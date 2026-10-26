import torch
import torch.nn as nn


class RadarObjectDensifier(nn.Module):
    """Predict virtual object points from sparse radar crops.

    The module encodes a fixed-size radar crop with a lightweight PointNet-style
    encoder and predicts a fixed set of normalized local XYZ offsets. The
    predicted points are later rescaled to the target 3D box size and optionally
    appended back to the radar point cloud during training.
    """

    def __init__(
        self,
        in_channels=7,
        hidden_channels=64,
        num_virtual_points=64,
    ):
        super().__init__()
        self.num_virtual_points = int(num_virtual_points)
        self.point_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, self.num_virtual_points * 3),
        )

    def forward(self, radar_crops, crop_mask):
        encoded = self.point_encoder(radar_crops)
        invalid_fill = torch.finfo(encoded.dtype).min
        encoded = encoded.masked_fill(~crop_mask.unsqueeze(-1), invalid_fill)
        pooled = encoded.max(dim=1).values
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

        local_points = self.global_mlp(pooled).view(-1, self.num_virtual_points, 3)
        return torch.tanh(local_points)
