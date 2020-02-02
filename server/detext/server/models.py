from __future__ import annotations

from django.contrib.auth.models import User
from django.db import models

import torch

import io

from scripts.models.mobilenet import MobileNet


class MathSymbol(models.Model):
    name = models.CharField(max_length=200)
    timestamp = models.DateTimeField()

    description = models.CharField(max_length=1000)
    latex = models.CharField(max_length=200, blank=True)
    image = models.BinaryField(editable=True, blank=True)

    def __str__(self):
        return f"{self.name}"

    @classmethod
    def get(cls, id) -> MathSymbol:
        return cls.objects.get(pk=id)


class TrainImage(models.Model):
    symbol = models.ForeignKey(MathSymbol, on_delete=models.CASCADE)
    image = models.BinaryField(editable=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True,
                             blank=True)
    locked = models.BooleanField(default=True)
    features = models.BinaryField(editable=True, blank=True)

    def __str__(self):
        return f"{self.symbol}"


class ClassificationModel(models.Model):
    model = models.BinaryField(editable=True)
    pytorch = models.BinaryField(editable=True, blank=True)
    timestamp = models.DateTimeField()
    accuracy = models.FloatField(default=0.9)

    def __str__(self):
        return f"{self.timestamp} - Accuracy: {self.accuracy}"

    @classmethod
    def get_latest(cls) -> ClassificationModel:
        return cls.objects.all().order_by('-timestamp').first()

    def to_pytorch(self) -> MobileNet:
        state_dict = torch.load(io.BytesIO(self.pytorch),
                                map_location=torch.device('cpu'))
        model = MobileNet(
            features=state_dict['mobilenet.classifier.1.bias'].shape[0],
            pretrained=False)
        model.load_state_dict(state_dict)

        return model
