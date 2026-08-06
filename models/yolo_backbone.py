import torch
import torch.nn as nn


class YOLOv10Backbone(nn.Module):
   

    def __init__(self, pretrained_weights: str = 'yolov10m.pt'):
        super().__init__()
        import os

       
        if pretrained_weights.endswith('.pt') and os.path.exists(pretrained_weights):
            ckpt = torch.load(pretrained_weights, map_location='cpu',
                              weights_only=False)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                self.model = ckpt['model'].float()
            elif isinstance(ckpt, nn.Module):
                self.model = ckpt.float()
            else:
                raise ValueError(
                    f"Unrecognised checkpoint format in {pretrained_weights}. "
                    "Expected a dict with 'model' key or a bare nn.Module.")
        else:
            try:
                from ultralytics import YOLO as _YOLO
                self.model = _YOLO('yolov10m.pt').model.float()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load backbone weights from '{pretrained_weights}': {exc}"
                )

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
                new_conv.weight.copy_(first_conv.weight[:, :2, :, :])
                if first_conv.bias is not None:
                    new_conv.bias = first_conv.bias
            if hasattr(self.model, 'model') and hasattr(self.model.model[0], 'conv'):
                self.model.model[0].conv = new_conv

        self.feature_maps: dict[str, torch.Tensor] = {}
        self.target_layer: str | None = None
        self._register_hooks()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        self.feature_maps = {}
        return self.model(x)

    def get_roi_features(
        self,
        boxes: torch.Tensor,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        
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
