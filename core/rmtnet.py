import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet50
from ViT import ViT
from VT import Transformer
from LMHSA import LightMutilHeadSelfAttention

class Stem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        return self.conv(x)

class RMTNet(nn.Module):
    def __init__(self, img_size=256, patch_size=16, num_classes=4, embed_dim=768, depth=12, num_heads=12, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super(RMTNet, self).__init__()

        # Stem module
        self.stem = Stem(in_channels=3, out_channels=64)

        # ResNet-50 Backbone (up to conv4_x)
        self.resnet = resnet50(pretrained=True)
        self.resnet_layers = nn.Sequential(
            self.stem,  # Use Stem as the initial layers
            self.resnet.layer1,
            Downsample(256),  # Add downsampling after each stage
            self.resnet.layer2,
            Downsample(512),  # Add downsampling after each stage
            self.resnet.layer3,
        )
        
        # Linear embedding
        self.linear_embedding = nn.Conv2d(1024, embed_dim, kernel_size=1)  # Adjust input channels if necessary

        # ViT module (Stage 1)
        self.vit = ViT(
            image_size=img_size // 16,  # Adjust to match the size after ResNet
            patch_size=1,
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
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        # Stem and ResNet-50 Backbone
        x = self.resnet_layers(x)  # [B, 1024, H/16, W/16]

        # Linear embedding
        x = self.linear_embedding(x)  # [B, embed_dim, H/16, W/16]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, N, embed_dim]

        # Stage 1: ViT
        x = self.vit(x)  # [B, N, embed_dim]

        # Downsample before Stage 2
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(np.sqrt(x.shape[1])))
        x = self.downsample(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Stage 2: VT
        x = self.vt(x)  # [B, N, embed_dim]

        # Downsample before Stage 3
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(np.sqrt(x.shape[1])))
        x = self.downsample(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Stage 3: LMHSA
        B, N, C = x.shape
        cls_token, x = x[:, 0], x[:, 1:]  # The class token is at the beginning
        H = W = int(np.sqrt(N))  # Removing '-1' to account for full N
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)  # [B, embed_dim, H, W]
        x = self.lmhsa(x)  # [B, embed_dim, H, W]

        # Classification head
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = torch.cat([cls_token, x], dim=1)  # Add the class token back
        x = self.fc(x)  # [B, num_classes]

        return x
