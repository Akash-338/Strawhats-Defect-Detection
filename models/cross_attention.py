import torch
import torch.nn as nn
import math

class CrossAttentionFusion(nn.Module):
    
    
    def __init__(self, visual_dim: int, morph_dim: int, d_model: int = 128, num_heads: int = 4):
       
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.W_Q = nn.Linear(visual_dim, d_model)
        self.W_K = nn.Linear(morph_dim, d_model)
        self.W_V = nn.Linear(morph_dim, d_model)
        
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
        self.norm = nn.LayerNorm(d_model + visual_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, f_visual: torch.Tensor, f_morph: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        
        squeezed = False
        if f_visual.dim() == 2:
            f_visual = f_visual.unsqueeze(0)
            f_morph = f_morph.unsqueeze(0)
            squeezed = True
            
        Q = self.W_Q(f_visual)  # (B, N, d_model)
        K = self.W_K(f_morph)   # (B, N, d_model)
        V = self.W_V(f_morph)   # (B, N, d_model)
        
        attn_output, _ = self.mha(Q, K, V, key_padding_mask=mask)
        
        fused = torch.cat([attn_output, f_visual], dim=-1)  # (B, N, d_model + visual_dim)
        
        output = self.norm(fused)
        
        if squeezed:
            output = output.squeeze(0)
            
        return output
