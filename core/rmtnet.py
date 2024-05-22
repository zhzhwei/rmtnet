import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ViT import ViT
from VT import Transformer
from LMHSA import LightMutilHeadSelfAttention

class RMTNet(nn.Module):
    def __init__(self, img_size=256, patch_size=16, num_classes=4, embed_dim=768, depth=12, num_heads=12, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super(RMTNet, self).__init__()
        
        # ViT module (Stage 1)
        self.vit = ViT(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        
        # VT module (Stage 2)
        self.vt = Transformer(
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
        # LMHSA module (Stage 3)
        self.lmhsa = LightMutilHeadSelfAttention(
            dim=embed_dim,
            num_heads=num_heads,
            features_size=img_size // patch_size,
            relative_pos_embeeding=True,
            no_distance_pos_embeeding=False,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=dropout,
            proj_drop=dropout,
            sr_ratio=1
        )
        
        # Final classification head
        self.fc = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, num_classes)
        )
    
    def forward(self, x):
        # Stage 1: ViT
        x = self.vit(x)
        
        # Stage 2: VT
        x = self.vt(x)
        
        # Stage 3: LMHSA
        B, N, C = x.shape
        cls_token, x = x[:, 0], x[:, 1:]  # The class token is at the beginning
        H = W = int(np.sqrt(N - 1))  # Subtract 1 to account for the class token
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.lmhsa(x)
        
        # Classification head
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = torch.cat([cls_token, x], dim=1)  # Add the class token back
        x = self.fc(x)
        
        return x
