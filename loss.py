import torch
import torch.nn.functional as F
from utils import concat_all_gather, get_rank

def clip_loss(image_features, text_features, logit_scale):
    """
    image_features: [B, embed_dim]
    text_features:  [B, embed_dim]
    logit_scale:  model.logit_scale  nn.Parameter, shape [1]
    """
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    all_image_features = concat_all_gather(image_features)                 # [world_size * B, embed_dim]
    all_text_features = concat_all_gather(text_features)                   # [world_size * B, embed_dim]

    logit_scale = logit_scale.exp()
    logits_i = logit_scale * image_features @ all_text_features.t()  # [B, world_size * B]
    logits_t = logit_scale * text_features @ all_image_features.t()  # [B, world_size * B]

    B = logits_i.size(0)
    labels = torch.arange(B, device=logits_i.device) + get_rank() * B

    loss_i2t = F.cross_entropy(logits_i, labels)
    loss_t2i = F.cross_entropy(logits_t, labels)

    loss = (loss_i2t + loss_t2i) / 2
    return loss


if __name__ == '__main__':
    from model.clip import CLIP
    text = torch.randint(1, 129, (16, 128))
    image = torch.randn(16, 3, 128, 128)
    model = CLIP(129, 129, 128, 4, 3, 64, 4, 7, 512)
    text_features, image_features, logit_scale = model(text, image)
    loss = clip_loss(image_features, text_features, logit_scale)
    print(loss.item())