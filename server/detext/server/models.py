from django.contrib.auth.models import User
from django.db import models


class MathSymbol(models.Model):
    name = models.CharField(max_length=200)
    timestamp = models.DateTimeField()

    description = models.CharField(max_length=1000)
    latex = models.CharField(max_length=200, blank=True)
    image = models.BinaryField(editable=True, blank=True)

    def __str__(self):
        return f"{self.name}"

class TrainImage(models.Model):
    symbol = models.ForeignKey(MathSymbol, on_delete=models.CASCADE)
    image = models.BinaryField(editable=True)
    timestamp = models.DateTimeField(auto_now_add = True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    locked = models.BooleanField(default=True)
    features = models.BinaryField(editable=True, blank=True)

    def __str__(self):
        return f"{self.symbol}"

class ClassificationModel(models.Model):
    model = models.BinaryField(editable=True)
    pytorch = models.BinaryField(editable=True, blank=True)
    timestamp = models.DateTimeField()
    accuracy = models.FloatField(default=0.9)

    # TODO: Add train/test accuracy, other infos?

    def __str__(self):
        return f"{self.timestamp} - Accuracy: {self.accuracy}"
