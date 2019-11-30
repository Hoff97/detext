import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim import lr_scheduler
from torchvision import transforms


class MobileNet(nn.Module):
    def __init__(self, features = 2, pretrained = True):
        super(MobileNet, self).__init__()

        self.num_features = features

        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=self.mobilenet.classifier[1].in_features, out_features=self.num_features, bias=True)
        )
        self.mobilenet.classifier = self.classifier

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mobilenet(x)

    def features(self, x):
        return self.mobilenet.features(x)

    def linear(self, x):
        return self.mobilenet.linear(x)

    def freeze(self):
        for p in self.mobilenet.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze(self):
        for p in self.mobilenet.parameters():
            p.requires_grad = True

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
