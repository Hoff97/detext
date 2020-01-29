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

    def set_classifier(self, classifier):
        self.classifier = classifier
        self.mobilenet.classifier = self.classifier

    def forward(self, x):
        return self.mobilenet(x)

    def features(self, x):
        return self.mobilenet.features(x)

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
