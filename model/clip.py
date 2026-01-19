from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from torch import nn
import torch


class CLIP(nn.Module):
    def __init__(self, vocab_size, max_len, img_size, patch_size, in_channels, d_model, num_heads, depth, embed_dim, eos_token_id=3):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, max_len, d_model, depth, num_heads, embed_dim, eos_token_id)
        self.image_encoder = ImageEncoder(img_size, patch_size, in_channels, d_model, num_heads, depth, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([1]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, text, image):
        """
        text shape: [B, T]
        image shape: [B, C, H, W]
        text_features shape: [B, embed_dim]
        image_features shape: [B, embed_dim]
        """
        text_features = self.text_encoder(text)
        image_features = self.image_encoder(image)
        return text_features, image_features, self.logit_scale


if __name__ == '__main__':
    text = torch.randint(1, 129, (1, 58)).cuda()
    image = torch.randn(1, 3, 480, 480).cuda()
    model = CLIP(16410, 58, 480, 8, 3, 1024, 16, 12, 512).cuda()
    text_features, image_features, logit_scale = model(text, image)
    total_parameters = 0
    for p in model.parameters():
        total_parameters += p.numel()
    print(f"total_parameters: {total_parameters}")
    print(f"text_features.shape: {text_features.shape}")
    print(f"image_features.shape: {image_features.shape}")
    print(f"logit_scale: {logit_scale.item()}")