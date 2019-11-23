import base64
import binascii
import io

from django.contrib.auth.models import Group, User
from PIL import Image
from rest_framework import mixins, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied, bad_request
from rest_framework.mixins import (CreateModelMixin, DestroyModelMixin,
                                   UpdateModelMixin)
from rest_framework.response import Response

from detext.server.models import ClassificationModel, MathSymbol, TrainImage
from detext.server.serializers import (ClassificationModelSerializer,
                                       MathSymbolSerializer,
                                       TrainImageSerializer)


class MathSymbolView(viewsets.ModelViewSet):
    queryset = MathSymbol.objects.all()
    serializer_class = MathSymbolSerializer
    ordering = ['timestamp']

    """
    Destroy a model instance.
    """
    def destroy(self, request, *args, **kwargs):
        if request.user.id is None:
            raise PermissionDenied({"message":"Can only delete class symbol when logged in"})
        instance = self.get_object()
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)

    """
    Create a model instance.
    """
    def create(self, request, *args, **kwargs):
        if request.user.id is None:
            request.data.image = None

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    """
    Update a model instance.
    """
    def update(self, request, *args, **kwargs):
        if request.user.id is None:
            raise PermissionDenied({"message":"Can only update class symbol when logged in"})

        partial = kwargs.pop('partial', False)
        instance = self.get_object()

        if 'image' in request.data:
            return bad_request(request)

        img = MathSymbol.objects.get(pk=instance.id).image
        instance.image = img

        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        if getattr(instance, '_prefetched_objects_cache', None):
            # If 'prefetch_related' has been applied to a queryset, we need to
            # forcibly invalidate the prefetch cache on the instance.
            instance._prefetched_objects_cache = {}

        return Response('Ok')

    @action(detail=True, methods=['put'])
    def image(self, request, pk=None):
        if pk == None:
            return bad_request(request, Exception('Primary key can not be null'))

        if 'image' not in request.data:
            return bad_request(request, Exception('Image needs to be sent with this request'))

        if request.user.id is None:
            raise PermissionDenied({"message":"Can only update class image when logged in"})

        symbol = MathSymbol.objects.get(pk=pk)
        symbol.image = base64.b64decode(request.data['image'])
        symbol.save()

        return Response('Ok')

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
