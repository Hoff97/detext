import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim import lr_scheduler
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

class CNNNet(nn.Module):
    def __init__(self, features = 2):
        super(CNNNet, self).__init__()

        self.features = features
        self.cnn_features = 53*53*10

        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 5, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(5, 10, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(self.cnn_features, 500)
        self.fc2 = nn.Linear(500, features)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = x.view(-1, self.cnn_features)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def freeze(self):
        pass

    def unfreeze(self):
        pass
