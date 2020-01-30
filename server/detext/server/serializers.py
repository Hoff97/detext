from rest_framework import serializers

from detext.server.models import ClassificationModel, MathSymbol, TrainImage


class MathSymbolSerializer(serializers.ModelSerializer):
    class Meta:
        model = MathSymbol
        fields = ['id', 'name', 'timestamp', 'description', 'latex', 'image']


class ClassificationModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClassificationModel
        fields = ['id', 'model', 'timestamp', 'accuracy']


class TrainImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainImage
        fields = ['id', 'symbol', 'image', 'timestamp', 'user']
