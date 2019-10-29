from django.contrib.auth.models import Group, User
from rest_framework import serializers

from detext.server.models import ClassificationModel, MathSymbol, TrainImage


class MathSymbolSerializer(serializers.ModelSerializer):
    class Meta:
        model = MathSymbol
        fields = ['name', 'timestamp']

class ClassificationModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClassificationModel
        fields = ['model', 'timestamp']

class TrainImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainImage
        fields = ['symbol', 'image', 'timestamp', 'user']
