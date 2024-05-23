import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ViT import ViT
from VT import Transformer
from LMHSA import LightMutilHeadSelfAttention

class Stem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class RMTNet(nn.Module):
    def __init__(self, img_size=128, patch_size=16, num_classes=4, embed_dim=768, depth=12, num_heads=12, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super(RMTNet, self).__init__()

        # Stem
        self.stem = Stem(in_channels=3, out_channels=64)
        
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
            emb_dropout=emb_dropout,
            channels = self.stem.out_channels
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
            features_size=img_size // patch_size, # height and width of the feature map
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
        x = self.stem(x)  # [B, 64, 128, 128]

        # Stage 1: ViT
        x = self.vit(x) # [B, 65, 768]
        
        # Stage 2: VT
        x = self.vt(x) # [B, 65, 768]
        
        # Stage 3: LMHSA
        B, N, C = x.shape
        cls_token, x = x[:, 0], x[:, 1:]  # The class token is at the beginning
        H = W = int(np.sqrt(N - 1))  # Subtract 1 to account for the class token
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W) # [B, 768, 8, 8]
        x = self.lmhsa(x) # [B, 768, 8, 8]
        
        # Classification head
        x = x.mean(dim=[2, 3])  # Global average pooling [B, 768]
        x = torch.cat([cls_token, x], dim=1)  # Add the class token back 
        x = self.fc(x) # [B, 4]
        
        return x