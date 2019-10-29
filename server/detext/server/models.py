from django.contrib.auth.models import User
from django.db import models


class MathSymbol(models.Model):
    name = models.CharField(max_length=200)
    timestamp = models.DateTimeField()

    def __str__(self):
        return f"{self.name}"

class TrainImage(models.Model):
    symbol = models.ForeignKey(MathSymbol, on_delete=models.CASCADE)
    image = models.BinaryField(editable=True)
    timestamp = models.DateTimeField()
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    locked = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.symbol}"

class ClassificationModel(models.Model):
    model = models.BinaryField(editable=True)
    timestamp = models.DateTimeField()

    def __str__(self):
        return f"{self.name}"
