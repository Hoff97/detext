from django.contrib.auth.models import Group, User
from rest_framework import mixins, permissions, viewsets

from detext.server.models import ClassificationModel, MathSymbol, TrainImage
from detext.server.serializers import (ClassificationModelSerializer,
                                       MathSymbolSerializer,
                                       TrainImageSerializer)


class MathSymbolView(viewsets.ModelViewSet):
    queryset = MathSymbol.objects.all()
    serializer_class = MathSymbolSerializer
    ordering = ['timestamp']

class ClassificationModelView(viewsets.ReadOnlyModelViewSet):
    queryset = ClassificationModel.objects.all()
    serializer_class = ClassificationModelSerializer
    ordering = ['timestamp']

class TrainImageView(viewsets.ModelViewSet):
    queryset = TrainImage.objects.all()
    serializer_class = TrainImageSerializer
