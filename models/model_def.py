import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0
from torchvision.models import ResNet18_Weights
from torchvision.models import EfficientNet_B0_Weights

#載入預訓練權重，classes=2 二分類
def get_resnet18(num_classes=2, pretrained=True):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_efficientnet_b0(num_classes=2, pretrained=True):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model
