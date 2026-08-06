import torch
import torch.nn as nn
import torch.nn.functional as F

class MorphologyClassificationHead(nn.Module):
   
    
    def __init__(self, in_features: int = 256, num_classes: int = 6, temperature: float = 1.0):
        
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        logits = self.classifier(x)
        
    
        scaled_logits = logits / self.temperature
        
        return scaled_logits
