import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim import lr_scheduler
from torchvision import transforms


class MobileNet(nn.Module):
    def __init__(self, features = 2, pretrained = True):
        super(MobileNet, self).__init__()

        self.features = features

        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=self.mobilenet.classifier[1].in_features, out_features=self.features, bias=True)
        )
        self.mobilenet.classifier = self.classifier

    def forward(self, x):
        return self.mobilenet(x)

    def freeze(self):
        for p in self.mobilenet.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze(self):
        for p in self.mobilenet.parameters():
            p.requires_grad = True

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
