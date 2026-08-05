import torch
import torch.nn as nn
import math

class CrossAttentionFusion(nn.Module):
    """Cross-Attention Fusion Module implementing paper Section 3.6.
    
    Fuses visual features from YOLO and morphology features.
    Q = W_Q · f_visual (from YOLO ROI features)
    K = W_K · f_morph (from morphology encoder)
    V = W_V · f_morph
    """
    
    def __init__(self, visual_dim: int, morph_dim: int, d_model: int = 128, num_heads: int = 4):
        """Initialize the cross-attention fusion module.
        
        Parameters
        ----------
        visual_dim : int
            Dimension of input visual features.
        morph_dim : int
            Dimension of input morphology features.
        d_model : int, optional
            Dimension of the attention model, by default 128
        num_heads : int, optional
            Number of attention heads, by default 4
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(visual_dim, d_model)
        self.W_K = nn.Linear(morph_dim, d_model)
        self.W_V = nn.Linear(morph_dim, d_model)
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
        # Layer normalization
        # Output is concatenated with original visual tokens, so dim is d_model + visual_dim
        self.norm = nn.LayerNorm(d_model + visual_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, f_visual: torch.Tensor, f_morph: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for cross-attention fusion.
        
        Parameters
        ----------
        f_visual : torch.Tensor
            Visual features of shape (B, N, visual_dim) or (N, visual_dim) if batch size is 1.
        f_morph : torch.Tensor
            Morphology features of shape (B, N, morph_dim) or (N, morph_dim).
        mask : torch.Tensor, optional
            Attention mask for variable number of detections, by default None
            
        Returns
        -------
        torch.Tensor
            Fused features of shape (B, N, d_model + visual_dim) or (N, d_model + visual_dim).
        """
        # Ensure batch dimension exists
        squeezed = False
        if f_visual.dim() == 2:
            f_visual = f_visual.unsqueeze(0)
            f_morph = f_morph.unsqueeze(0)
            squeezed = True
            
        # Project inputs
        Q = self.W_Q(f_visual)  # (B, N, d_model)
        K = self.W_K(f_morph)   # (B, N, d_model)
        V = self.W_V(f_morph)   # (B, N, d_model)
        
        # Apply multi-head attention
        attn_output, _ = self.mha(Q, K, V, key_padding_mask=mask)
        
        # Concatenate fused embedding with original visual tokens
        fused = torch.cat([attn_output, f_visual], dim=-1)  # (B, N, d_model + visual_dim)
        
        # Layer normalization
        output = self.norm(fused)
        
        # Remove batch dim only if we added it
        if squeezed:
            output = output.squeeze(0)
            
        return output
