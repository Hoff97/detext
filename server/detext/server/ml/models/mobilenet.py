import io

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class MobileNet(nn.Module):
    def __init__(self, features=2, pretrained=True, **kwargs):
        super(MobileNet, self).__init__()

        self.num_features = features

        self.mobilenet = models.mobilenet_v2(pretrained=pretrained, **kwargs)
        print(self.mobilenet.classifier[1].in_features)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=self.mobilenet.classifier[1].in_features,
                      out_features=self.num_features, bias=True)
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

    def freeze(self):
        for p in self.mobilenet.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze(self):
        for p in self.mobilenet.parameters():
            p.requires_grad = True

    @classmethod
    def from_file(cls, file):
        state_dict = torch.load(file)
        n_features = state_dict['mobilenet.classifier.1.bias'].shape[0]
        model = MobileNet(features=n_features)
        model.load_state_dict(state_dict)
        return model

    def to_onnx(self):
        byte_arr = io.BytesIO()
        dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
        torch.onnx.export(self.to("cpu"), dummy_input, byte_arr)
        return byte_arr


preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
