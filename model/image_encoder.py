import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):       # [B, C, H, W]
        x = self.proj(x)        # [B, d_model, H/patch, W/patch]
        x = x.flatten(2)        # [B, d_model, num_patches]
        x = x.transpose(1, 2)   # [B, num_patches, d_model]
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, num_patches, d_model = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, num_patches, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)                           # [3, B, num_heads, num_patches, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]                           # [B, num_heads, num_patches, d_k]
        attn = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, num_heads, num_patches, num_patches]
        attn = attn.softmax(dim=-1)
        out = attn @ v                                             # [B, num_heads, num_patches, d_k]
        out = out.transpose(1, 2).reshape(B, num_patches, d_model) # [B, num_patches, d_model]
        out = self.W_o(out)                                        # [B, num_patches, d_model]
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model, num_heads, depth, embed_dim, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches+1, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.image_projection = nn.Linear(d_model, embed_dim, bias=False)

    def forward(self, x):
        """
        input shape: [B, C, H, W]
        return shape: [B, embed_dim]
        """
        x = self.patch_embed(x)                                 # [B, num_patches, d_model]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed                                  # [B, num_patches, d_model]
        for block in self.blocks:
            x = block(x)                                        # [B, num_patches, d_model]
        x = self.norm(x)                                        # [B, num_patches, d_model]
        cls_token = x[:, 0]                                     # [B, d_model]
        image_features = self.image_projection(cls_token)       # [B, embed_dim]

        return image_features


if __name__ == "__main__":
    image = torch.randn(16, 3, 128, 128).cuda()
    model = ImageEncoder(128, 4, 3, 64, 4, 7, 512).cuda()
    image_features = model(image)
    print(f"input shape: {image.shape}")
    print(f"output shape: {image_features.shape}")




