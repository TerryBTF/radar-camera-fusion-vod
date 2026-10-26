import torch
from torch import nn


class PointGatedFusion(nn.Module):
    def __init__(
        self,
        radar_dim,
        image_dim,
        hidden_dim=None,
        image_scale_dims=None,
        context_dim=0,
        structured_fusion=False,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = radar_dim

        self.structured_fusion = structured_fusion
        self.context_dim = context_dim
        self.image_scale_dims = image_scale_dims or [image_dim]

        if self.structured_fusion:
            self.image_proj_layers = nn.ModuleList(
                nn.Linear(scale_dim, radar_dim) for scale_dim in self.image_scale_dims
            )
            self.scale_gate_mlps = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(radar_dim * 2 + context_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, radar_dim),
                )
                for _ in self.image_scale_dims
            )
        else:
            self.image_proj = nn.Linear(image_dim, radar_dim)
            self.gate_mlp = nn.Sequential(
                nn.Linear(radar_dim * 2 + context_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, radar_dim),
            )

    def _append_context(self, gate_input, context_features):
        if self.context_dim <= 0:
            return gate_input
        if context_features is None:
            raise ValueError("context_features are required when context_dim > 0.")
        return torch.cat([gate_input, context_features], dim=-1)

    def forward(self, radar_features, image_features, context_features=None):
        if not self.structured_fusion:
            projected_image = self.image_proj(image_features)
            gate_input = torch.cat([radar_features, projected_image], dim=-1)
            gate_input = self._append_context(gate_input, context_features)
            gate = torch.sigmoid(self.gate_mlp(gate_input))
            fused = radar_features + gate * projected_image
            return fused, gate

        image_splits = torch.split(image_features, self.image_scale_dims, dim=-1)
        fused = radar_features
        gates = []
        for image_split, image_proj, gate_mlp in zip(
            image_splits,
            self.image_proj_layers,
            self.scale_gate_mlps,
        ):
            projected_image = image_proj(image_split)
            gate_input = torch.cat([radar_features, projected_image], dim=-1)
            gate_input = self._append_context(gate_input, context_features)
            scale_gate = torch.sigmoid(gate_mlp(gate_input))
            fused = fused + scale_gate * projected_image
            gates.append(scale_gate)

        gate = torch.stack(gates, dim=0).mean(dim=0)
        return fused, gate
