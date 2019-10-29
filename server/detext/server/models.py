from django.db import models


class MathSymbol(models.Model):
    name = models.CharField(max_length=200)
    order = models.IntegerField()
