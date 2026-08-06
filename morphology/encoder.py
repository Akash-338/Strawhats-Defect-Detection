import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MorphologyEncoder(nn.Module):
    
    def __init__(self):
        super(MorphologyEncoder, self).__init__()
        
        self.fc1 = nn.Linear(11, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 128)
        self.layernorm = nn.LayerNorm(128)

        self.reconstruction_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 11)
        )

        self._init_weights()

    def _init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
            
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        f_morph = self.layernorm(out)
        
        return f_morph

    def decode(self, f_morph: torch.Tensor) -> torch.Tensor:
        
        return self.reconstruction_head(f_morph)
