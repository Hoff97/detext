import torch.nn as nn
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


class CNNNet(nn.Module):
    def __init__(self, features=2):
        super(CNNNet, self).__init__()

        self.features = features
        self.cnn_features = 24*24*10

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
        self.cnn3 = nn.Sequential(
            nn.Conv2d(10, 10, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.cnn_features, 100),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(100, features),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = x.view(-1, self.cnn_features)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def freeze(self):
        pass

    def unfreeze(self):
        pass
