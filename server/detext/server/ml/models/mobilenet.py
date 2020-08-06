import io

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F


class TestTimeDropout(nn.Module):
    __constants__ = ['p', 'inplace']

    def __init__(self, p=0.5, inplace=False):
        super(TestTimeDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self):
        return 'p={}, inplace={}'.format(self.p, self.inplace)

    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)


mobilenet_dropout = 0.2


class MobileNet(nn.Module):
    def __init__(self, features=2, pretrained=True, test_time_dropout=False,
                 estimate_variane=False, **kwargs):
        super(MobileNet, self).__init__()

        self.num_features = features

        self.mobilenet = models.mobilenet_v2(pretrained=pretrained, **kwargs)

        dropout_module = nn.Dropout
        if test_time_dropout:
            dropout_module = TestTimeDropout

        self.classifier = nn.Sequential(
            dropout_module(p=mobilenet_dropout, inplace=False),
            nn.Linear(in_features=self.mobilenet.classifier[1].in_features,
                      out_features=self.num_features, bias=True)
        )
        self.mobilenet.classifier = self.classifier

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.frozen = False

        self.estimate_variane = estimate_variane

    def set_classifier(self, classifier):
        self.classifier = classifier
        self.mobilenet.classifier = self.classifier

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = x.mean([2, 3])

        if self.estimate_variane:
            # B = x.shape[0]
            B = 1            # This is only done so shape is not used
            D = 1280

            mu = self.mobilenet.classifier(x)

            C = self.num_features

            weights = self.mobilenet.classifier[1].weight\
                .reshape((1, self.num_features, -1)).repeat((B, 1, 1))

            cov = x.reshape((-1, D, 1))
            cov = torch.bmm(weights, weights.transpose(2, 1) * cov)
            var = torch.bmm(weights, x.reshape((-1, D, 1))).reshape((-1, C))

            # mu, cov are the mean, covariance after the last layer

            exp = torch.exp(mu)
            exp_sum = torch.sum(exp, dim=1, keepdim=True).repeat((1, C))

            var_s = (exp*(exp_sum - exp)/(exp_sum*exp_sum)) * var
            # This approximates the variance after the softmax

            return torch.stack((mu, var_s))

        x = self.mobilenet.classifier(x)
        return x

    def features(self, x):
        return self.mobilenet.features(x)

    def freeze(self):
        self.frozen = True
        for p in self.mobilenet.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze(self):
        self.frozen = True
        for p in self.mobilenet.parameters():
            p.requires_grad = True

    @classmethod
    def from_file(cls, file, **kwargs):
        state_dict = torch.load(file)
        n_features = state_dict['mobilenet.classifier.1.bias'].shape[0]
        model = MobileNet(features=n_features, **kwargs)
        model.load_state_dict(state_dict)
        return model

    def to_file(self, file):
        torch.save(self.state_dict(), file)

    def to_onnx(self):
        byte_arr = io.BytesIO()
        dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
        torch.onnx.export(self.to("cpu"), dummy_input, byte_arr)
        return byte_arr


preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
