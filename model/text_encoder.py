import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)                                                  # [B, T, 3*d_model]
        qkv = qkv.view(B, T, 3, self.n_heads, self.d_k)                         # [B, T, 3, n_heads, d_k]
        qkv = qkv.permute(2, 0, 3, 1, 4)                                        # [3, B, n_heads, T, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]                                        # [B, n_heads, T, d_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k ** 0.5    # [B, n_heads, T, T]
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_probs = F.softmax(attn_scores, dim=-1)                             # [B, n_heads, T, T]
        attn_out = torch.matmul(attn_probs, v)                                  # [B, n_heads, T, d_k]
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, self.d_model)
        out = self.out_proj(attn_out)                                           # [B, T, d_model]
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedMultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x, attn_mask=None):
        attn = self.attn(x, attn_mask=attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_layers, n_heads, embed_dim, eos_token_id=3):
        super().__init__()
        self.eos_token_id = eos_token_id

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.empty(max_len, d_model))

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])

        self.ln_final = nn.LayerNorm(d_model)
        self.text_projection = nn.Linear(d_model, embed_dim, bias=False)

        nn.init.normal_(self.pos_embedding, std=0.01)

    def build_causal_mask(self, seq_len, device):
        return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(device)

    def forward(self, input_ids):
        """
        input shape: [B, T]
        return shape: [B, embed_dim]
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_embedding(input_ids)                  # [B, T, d_model]
        x = x + self.pos_embedding[:T]                       # [B, T, d_model]

        attn_mask = self.build_causal_mask(T, device)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_final(x)

        eos_pos = (input_ids == self.eos_token_id).int().argmax(dim=1)
        text_features = x[torch.arange(B), eos_pos]          # [B, d_model]

        text_features = self.text_projection(text_features)  # [B, embed_dim]

        return text_features


if __name__ == "__main__":
    text = torch.randint(1, 129, (16, 128)).cuda()
    eos_token = torch.zeros(16, 1, dtype=torch.int).cuda()
    text = torch.cat([text, eos_token], dim=1)
    textencoder = TextEncoder(129, 129, 256, 2, 4, 512).cuda()
    text_features = textencoder(text)
    print(text_features.shape)



