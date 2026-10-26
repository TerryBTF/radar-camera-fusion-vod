import torch
from torch import nn


class TemporalRadarEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim=32,
        feature_indices=(3, 4, 5, 6),
        residual_scale=1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_indices = tuple(feature_indices)
        self.residual_scale = residual_scale

        feature_dim = len(self.feature_indices)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, points):
        if points.numel() == 0 or points.shape[1] < max(self.feature_indices) + 1:
            return points

        encoded = points.clone()
        temporal_slice = encoded[:, self.feature_indices]
        time_feature = temporal_slice[:, -1:].clone()
        vr = temporal_slice[:, 1:2] if temporal_slice.shape[1] > 1 else time_feature.new_zeros(time_feature.shape)
        vr_comp = temporal_slice[:, 2:3] if temporal_slice.shape[1] > 2 else time_feature.new_zeros(time_feature.shape)
        temporal_context = torch.cat([temporal_slice, time_feature * vr, time_feature * vr_comp, vr - vr_comp], dim=1)
        residual = self.mlp(temporal_context)
        encoded[:, self.feature_indices] = temporal_slice + self.residual_scale * residual
        return encoded
