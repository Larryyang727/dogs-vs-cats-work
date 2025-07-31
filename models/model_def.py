import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b0

def get_resnet18(num_classes=2):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_efficientnet_b0(num_classes=2):
    model = efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model