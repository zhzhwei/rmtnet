import torch
import torch.nn as nn

class Stem(nn.Module):
    def __init__(self, input_channels, patch_size, embed_dim):
        super(Stem, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.flatten = nn.Flatten(2, 3)
        self.linear = nn.Linear(patch_size * patch_size * input_channels, embed_dim)

    def forward(self, x):
        # Convert image to patches and apply convolution
        x = self.conv(x)  # [batch_size, embed_dim, H/patch_size, W/patch_size]
        # Flatten patches
        x = self.flatten(x)  # [batch_size, embed_dim, (H/patch_size) * (W/patch_size)]
        # Transpose and apply linear transformation
        x = x.transpose(1, 2)  # [batch_size, (H/patch_size) * (W/patch_size), embed_dim]
        x = self.linear(x)  # [batch_size, (H/patch_size) * (W/patch_size), embed_dim]
        return x

# Example usage:
# input_channels = 3 (for RGB images)
# patch_size = 16 (each patch is 16x16)
# embed_dim = 768 (dimension of the embedding vector for each patch)

input_channels = 3
patch_size = 16
embed_dim = 768
stem = Stem(input_channels, patch_size, embed_dim)

# Example input: a batch of 4 images, each 256x256 with 3 channels
input_tensor = torch.randn(4, 3, 256, 256)
output = stem(input_tensor)
print(output.shape)  # Should print: torch.Size([4, 256, 768])
