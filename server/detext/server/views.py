import base64
import binascii
import io
import urllib

from django.conf import settings
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
import scripts.models.mobilenet as mm
from scripts.models.mobilenet import MobileNet
import torch


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
            request.data['image'] = None

        code = request.data['latex']

        if 'image' not in request.data or request.data['image'] == None or request.data['image'] == '':
            url = settings.TEXSVG_URL + '?latex=' + code
            svg = urllib.request.urlopen(url).read()

            svg = base64.b64encode(svg).decode('utf-8')
            pre = 'data:image/svg+xml;base64,'
            svg = (pre + svg).encode('utf-8')
            request.data['image'] = svg

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
        img = Image.frombytes('RGBA', (request.data.pop('width'), request.data.pop('height')), imgBytes).convert('L')

        byteArr = io.BytesIO()
        img.save(byteArr, format='png')
        byteArr = byteArr.getvalue()
        request.data['image'] = base64.encodebytes(byteArr).decode('utf-8')

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        train_image = serializer.save()

        self.update_features(train_image, img)

        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def update_features(self, train_image, img):
        with torch.no_grad():
            latest_model = ClassificationModel.objects.all().order_by('-timestamp').first()
            old_classes = MathSymbol.objects.all().filter(timestamp__lte=latest_model.timestamp)
            model = mm.MobileNet(features=len(old_classes), pretrained=False)
            model.load_state_dict(torch.load(io.BytesIO(latest_model.pytorch)))
            model = model.eval()

            img = mm.preprocess(img)
            img = img.repeat((3,1,1))
            img = img.reshape((1,img.shape[0], img.shape[1], img.shape[2]))

            features = model.features(img)
            features = features.mean([2, 3])
            byte_f = io.BytesIO()
            torch.save(features, byte_f)

            train_image.features = byte_f.getvalue()
            train_image.save()
