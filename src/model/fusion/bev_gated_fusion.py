import torch
from torch import nn


class BEVGatedFusion(nn.Module):
    def __init__(self, radar_channels, image_channels, hidden_channels=128):
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.Conv2d(image_channels, radar_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(radar_channels),
            nn.ReLU(inplace=True),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(radar_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, radar_channels, kernel_size=1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(radar_channels, radar_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(radar_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, radar_bev, image_bev):
        projected_image = self.image_proj(image_bev)
        gate = torch.sigmoid(self.gate_net(torch.cat([radar_bev, projected_image], dim=1)))
        fused = radar_bev + gate * projected_image
        fused = self.refine(fused)
        return fused, gate
