from torch import nn
import torch
import math

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim):
        """
        Args:
            patch_size (int): size of the patches to be cut out of the given image
            embed_dim (int): size of embedding dimension
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        """
        Takes the data and cuts it into patches, then projects the resulting tensor into embedding dimension

        Args:
            x (Tensor): data of shape [B, 3, 32, 32]
        Returns:
            Tensor of shape [B, num_patches, embed_dim]
        """
        x = self.conv(x) # [B, embed_dim, H1, W1]
        x = x.flatten(2) # [B, embed_dim, num_patches]
        x = x.transpose(1, 2) # [B, num_patches, embed_dim]
        
        return x
    

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, grid_size_h, grid_size_w):
        super().__init__()
        pe = self._generate_sinusoids_2d(embed_dim, grid_size_h, grid_size_w)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe
    
    def _generate_sinusoids_2d(self, embed_dim, H, W):
        assert embed_dim % 2 == 0, "Embedding dimension size must be even"

        dim_h = embed_dim // 2
        dim_w = embed_dim // 2

        div_term = torch.exp(
            torch.arange(0, dim_h, 2) * (-math.log(10000.0) / dim_h)
        )

        pos_h = torch.arange(H).unsqueeze(1)
        pos_w = torch.arange(W).unsqueeze(1)

        pe_h = torch.zeros(H, dim_h)
        pe_h[:, 0::2] = torch.sin(pos_h * div_term)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term)

        pe_w = torch.zeros(W, dim_w)
        pe_w[:, 0::2] = torch.sin(pos_w * div_term)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term)

        # NE DOPISANO DOPISAT BISTRO + SROCHNO
