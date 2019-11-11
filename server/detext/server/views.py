import base64
import binascii
import io

from django.contrib.auth.models import Group, User
from PIL import Image
from rest_framework import mixins, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.mixins import CreateModelMixin
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

    def create(self, request, *args, **kwargs):
        imgB64 = request.data['image']
        imgBytes = base64.decodebytes(imgB64.encode())
        img = Image.frombytes('RGBA', (request.data.pop('width'), request.data.pop('height')), imgBytes)

        byteArr = io.BytesIO()
        img.save(byteArr, format='png')
        byteArr = byteArr.getvalue()
        request.data['image'] = base64.encodebytes(byteArr).decode('utf-8')

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        serializer.save()

        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
