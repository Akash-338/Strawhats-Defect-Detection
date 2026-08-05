import torch
import torch.nn as nn


class YOLOv10Backbone(nn.Module):
    """Wrapper that loads a trained YOLOv10 checkpoint purely as a frozen
    PyTorch feature extractor — bypasses Ultralytics Trainer entirely to
    prevent accidental COCO8 fine-tuning on backbone init.

    Feature extraction strategy
    ---------------------------
    A forward hook is registered on the deepest SPPF block in the model.
    After each backbone forward pass the captured feature map is available
    via ``self.feature_maps[self.target_layer]``.  ROI features are then
    pooled from this map with ``get_roi_features()``.
    """

    def __init__(self, pretrained_weights: str = 'yolov10m.pt'):
        super().__init__()
        import os

        # ── Load the model weights WITHOUT touching Ultralytics Trainer ────
        # We load the raw checkpoint and grab ckpt['model'] (a DetectionModel)
        # directly.  Creating YOLO('yolov10m.pt') here would trigger an
        # Ultralytics training run on coco8 with default args — avoid that.
        if pretrained_weights.endswith('.pt') and os.path.exists(pretrained_weights):
            ckpt = torch.load(pretrained_weights, map_location='cpu',
                              weights_only=False)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                # Standard Ultralytics checkpoint — model is already trained
                self.model = ckpt['model'].float()
            elif isinstance(ckpt, nn.Module):
                # Checkpoint IS the model (rare but possible)
                self.model = ckpt.float()
            else:
                raise ValueError(
                    f"Unrecognised checkpoint format in {pretrained_weights}. "
                    "Expected a dict with 'model' key or a bare nn.Module.")
        else:
            # Fallback: load a clean yolov10m without running Trainer
            try:
                # Use Ultralytics only to get the architecture, then detach
                from ultralytics import YOLO as _YOLO
                self.model = _YOLO('yolov10m.pt').model.float()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load backbone weights from '{pretrained_weights}': {exc}"
                )

        # ── Adapt first Conv for 2-channel input (grayscale + attention) ───
        first_conv = None
        for _name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break

        if first_conv is not None and first_conv.in_channels == 3:
            new_conv = nn.Conv2d(
                in_channels=2,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=(first_conv.bias is not None),
            )
            with torch.no_grad():
                # Copy the first 2 channels of the pretrained weights
                new_conv.weight.copy_(first_conv.weight[:, :2, :, :])
                if first_conv.bias is not None:
                    new_conv.bias = first_conv.bias
            # Replace in the model graph
            if hasattr(self.model, 'model') and hasattr(self.model.model[0], 'conv'):
                self.model.model[0].conv = new_conv

        # ── Register SPPF hook for intermediate features ────────────────────
        self.feature_maps: dict[str, torch.Tensor] = {}
        self.target_layer: str | None = None
        self._register_hooks()

    # ──────────────────────────────────────────────────────────────────────────
    def _register_hooks(self):
        """Attach a forward hook to every SPPF block (last one wins as target)."""
        def _make_hook(name: str):
            def hook(_module, _inp, output):
                self.feature_maps[name] = output
            return hook

        for idx, (_name, module) in enumerate(self.model.named_modules()):
            if module.__class__.__name__ == 'SPPF':
                key = f'feat_{idx}'
                module.register_forward_hook(_make_hook(key))
                self.target_layer = key   # keep updating → last SPPF wins

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone forward pass; feature maps are stored in
        ``self.feature_maps`` via the registered hooks.

        Parameters
        ----------
        x : Tensor  shape (B, 2, H, W)
        """
        self.feature_maps = {}
        return self.model(x)

    # ──────────────────────────────────────────────────────────────────────────
    def get_roi_features(
        self,
        boxes: torch.Tensor,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        """Pool per-box features from the last SPPF feature map.

        Parameters
        ----------
        boxes : Tensor  shape (N, 4)  normalised (x1, y1, x2, y2) in [0, 1]
        batch_idx : int  which item in the batch to pool from

        Returns
        -------
        Tensor  shape (N, C)
        """
        import torchvision

        if not self.feature_maps or self.target_layer not in self.feature_maps:
            raise RuntimeError(
                "No feature maps available — run forward() first.")

        feat = self.feature_maps[self.target_layer]   # (B, C, H, W)
        _B, C, H, W = feat.shape

        if boxes.shape[0] == 0:
            return torch.empty((0, C), device=feat.device)

        scaled = boxes.clone().float()
        scaled[:, [0, 2]] *= W
        scaled[:, [1, 3]] *= H

        batch_col = torch.full(
            (boxes.shape[0], 1), batch_idx,
            device=boxes.device, dtype=scaled.dtype,
        )
        roi_boxes = torch.cat([batch_col, scaled], dim=1)

        pooled = torchvision.ops.roi_align(
            feat, roi_boxes, output_size=(1, 1), spatial_scale=1.0
        )
        return pooled.squeeze(-1).squeeze(-1)   # (N, C)
