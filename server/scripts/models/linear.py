import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, features=2, classes=2):
        super(LinearModel, self).__init__()

        self.num_features = features
        self.num_classes = classes

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=self.num_features,
                      out_features=self.num_classes, bias=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(x)
