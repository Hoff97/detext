from django.contrib.auth.models import Group, User
from rest_framework import mixins, permissions, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from detext.server.models import ClassificationModel, MathSymbol, TrainImage
from detext.server.serializers import (ClassificationModelSerializer,
                                       MathSymbolSerializer,
                                       TrainImageSerializer)


class MathSymbolView(viewsets.ModelViewSet):
    queryset = MathSymbol.objects.all()
    serializer_class = MathSymbolSerializer
    ordering = ['timestamp']

class ClassificationModelView(viewsets.ViewSet):
    queryset = ClassificationModel.objects.all()
    serializer_class = ClassificationModelSerializer

    @action(detail=False)
    def latest(self, request):
        latest = ClassificationModel.objects.all().order_by('-timestamp').first()

        serializer = ClassificationModelSerializer(latest, many=False)
        return Response(serializer.data)

class TrainImageView(viewsets.ModelViewSet):
    queryset = TrainImage.objects.all()
    serializer_class = TrainImageSerializer
