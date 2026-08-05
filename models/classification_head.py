import torch
import torch.nn as nn
import torch.nn.functional as F

class MorphologyClassificationHead(nn.Module):
    """Morphology-refined classifier.
    
    Takes fused embeddings and outputs morphology-informed predictions.
    Architecture: Linear(256,128) → ReLU → Dropout(0.3) → Linear(128,6) → Softmax
    """
    
    def __init__(self, in_features: int = 256, num_classes: int = 6, temperature: float = 1.0):
        """Initialize the classification head.
        
        Parameters
        ----------
        in_features : int, optional
            Dimension of fused input embeddings, by default 256
        num_classes : int, optional
            Number of defect classes, by default 6
        temperature : float, optional
            Temperature scaling for confidence calibration, by default 1.0
        """
        super().__init__()
        
        self.temperature = temperature
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Fused features of shape (N, in_features) or (B, N, in_features).
            
        Returns
        -------
        torch.Tensor
            Raw logits of shape (..., num_classes).
            NOTE: Do NOT apply softmax here — CrossEntropyLoss expects raw logits.
            Use F.softmax() only at inference time for probabilities.
        """
        # Get raw logits
        logits = self.classifier(x)
        
        # Apply temperature scaling (for inference calibration)
        scaled_logits = logits / self.temperature
        
        return scaled_logits
