import torch
import torch.nn as nn


class RadarBEVSelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        reduction: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.reduction = max(1, int(reduction))
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool = (
            nn.AvgPool2d(kernel_size=self.reduction, stride=self.reduction)
            if self.reduction > 1
            else nn.Identity()
        )
        hidden_channels = int(in_channels * ffn_ratio)
        self.norm2 = nn.GroupNorm(1, in_channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        normalized = self.norm1(x)
        query = normalized.flatten(2).transpose(1, 2)

        pooled = self.pool(normalized)
        key_value = pooled.flatten(2).transpose(1, 2)

        attn_out, _ = self.attn(query, key_value, key_value, need_weights=False)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, channels, height, width)

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x
