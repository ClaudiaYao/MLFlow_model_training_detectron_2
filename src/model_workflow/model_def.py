import torch
import torch.nn as nn

# torchvision's DINOv2 implementation provides pretrained backbones that output feature embeddings. We can use these embeddings as input to a simple classifier head for our multi-label classification task.
class DinoV2Classifier(nn.Module):
    def __init__(self, num_classes=3, backbone_name="dinov2_vits14"):
        super(DinoV2Classifier, self).__init__()

        # Load pretrained DINOv2 backbone
        backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)

        # Freeze backbone
        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x