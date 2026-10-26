from torch import nn
import logging

from torchvision.models import ResNet18_Weights, resnet18

logger = logging.getLogger(__name__)


class ResNet18Layer3(nn.Module):
    def __init__(
        self,
        pretrained=True,
        trainable_stage="none",
        freeze_bn=False,
        feature_layer="layer3",
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        
        # Try to load with pretrained weights, fallback to no weights if network unavailable
        try:
            backbone = resnet18(weights=weights)
            if pretrained:
                logger.info("✅ ResNet18 loaded with pretrained weights")
        except Exception as e:
            logger.warning(f"⚠️ Could not load pretrained weights ({type(e).__name__}: {str(e)[:100]})")
            logger.warning("   Loading ResNet18 with random initialization instead")
            backbone = resnet18(weights=None)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.feature_layer = feature_layer
        if self.feature_layer == "layer3":
            self.out_channels = 256
        elif self.feature_layer == "layer2_layer3":
            self.out_channels = 128 + 256
        else:
            raise ValueError(
                f"Unsupported feature_layer: {self.feature_layer}. "
                "Expected 'layer3' or 'layer2_layer3'."
            )
        self.trainable_stage = trainable_stage
        self.freeze_bn = freeze_bn

        self._freeze_parameters()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

        if self.trainable_stage == "layer3":
            for param in self.layer3.parameters():
                param.requires_grad = True
            if self.freeze_bn:
                self._freeze_batch_norm(self.layer3)

    def _freeze_batch_norm(self, module):
        for submodule in module.modules():
            if isinstance(submodule, nn.BatchNorm2d):
                submodule.eval()
                if submodule.weight is not None:
                    submodule.weight.requires_grad = False
                if submodule.bias is not None:
                    submodule.bias.requires_grad = False

    def forward(self, images):
        x = self.stem(images)
        x = self.layer1(x)
        layer2_feat = self.layer2(x)
        layer3_feat = self.layer3(layer2_feat)
        if self.feature_layer == "layer3":
            return layer3_feat
        return layer2_feat, layer3_feat

    def train(self, mode=True):
        super().train(mode)

        self.stem.eval()
        self.layer1.eval()
        self.layer2.eval()

        if self.trainable_stage == "layer3":
            self.layer3.train(mode)
            if self.freeze_bn:
                self._freeze_batch_norm(self.layer3)
        else:
            self.layer3.eval()

        return self
