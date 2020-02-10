import base64
import io
import urllib

import matplotlib.pyplot as plt
import numpy as np
import torch
from django.conf import settings
from django.db.models import Count
from django.http import HttpResponse
from PIL import Image
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied, bad_request
from rest_framework.response import Response
from tqdm import tqdm

import detext.server.ml.models.mobilenet as mm
from detext.server.ml.train_classifier import train_classifier
from detext.server.models import ClassificationModel, MathSymbol, TrainImage
from detext.server.serializers import (ClassificationModelSerializer,
                                       MathSymbolSerializer,
                                       TrainImageSerializer)
from detext.server.util.util import timeit


class MathSymbolView(viewsets.ModelViewSet):
    queryset = MathSymbol.objects.all()
    serializer_class = MathSymbolSerializer
    ordering = ['timestamp']

    """
    Destroy a model instance.
    """
    def destroy(self, request, *args, **kwargs):
        if request.user.id is None:
            raise PermissionDenied({
                "message": "Can only delete class symbol when logged in"
            })
        instance = self.get_object()
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)

    """
    Create a model instance.
    """
    def create(self, request, *args, **kwargs):
        req_data = request.data.copy()

        if request.user.id is None:
            req_data['image'] = None

        code = req_data['latex']

        if 'image' not in req_data or req_data['image'] is None\
                or req_data['image'] == '':
            svg = self.get_latex_svg(code)
            req_data['image'] = svg

        serializer = self.get_serializer(data=req_data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED,
                        headers=headers)

    """
    Update a model instance.
    """
    def update(self, request, *args, **kwargs):
        if request.user.id is None:
            raise PermissionDenied({
                "message": "Can only update class symbol when logged in"
            })

        partial = kwargs.pop('partial', False)
        instance = self.get_object()

        if 'image' in request.data:
            err_msg = 'Image should not be contained in update request'
            return bad_request(request, Exception(err_msg))

        img = MathSymbol.objects.get(pk=instance.id).image
        instance.image = img

        serializer = self.get_serializer(instance, data=request.data,
                                         partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        if instance.image is None or len(instance.image) == 0:
            instance.image = self.get_latex_svg(instance.latex)
            instance.save()

        if getattr(instance, '_prefetched_objects_cache', None):
            # If 'prefetch_related' has been applied to a queryset, we need to
            # forcibly invalidate the prefetch cache on the instance.
            instance._prefetched_objects_cache = {}

        return Response('Ok')

    def get_latex_svg(self, code):
        url = settings.TEXSVG_URL + '?latex=' + code
        svg = urllib.request.urlopen(url).read()

        svg = base64.b64encode(svg).decode('utf-8')
        pre = 'data:image/svg+xml;base64,'
        svg = (pre + svg).encode('utf-8')
        return svg

    @action(detail=True, methods=['put'])
    def image(self, request, pk=None):
        if pk is None:
            return bad_request(request,
                               Exception('Primary key can not be null'))

        if 'image' not in request.data:
            err_msg = 'Image needs to be sent with this request'
            return bad_request(request, Exception(err_msg))

        if request.user.id is None:
            raise PermissionDenied({
                "message": "Can only update class image when logged in"
            })

        try:
            symbol = MathSymbol.objects.get(pk=pk)
            symbol.image = base64.b64decode(request.data['image'])
            symbol.save()

            return Response('Ok')
        except MathSymbol.DoesNotExist as e:
            return bad_request(request, e)


class ClassificationModelView(viewsets.ViewSet):
    queryset = ClassificationModel.objects.all()
    serializer_class = ClassificationModelSerializer

    @action(detail=False)
    def latest(self, request):
        latest = ClassificationModel.objects.all()\
            .order_by('-timestamp').first()

        serializer = ClassificationModelSerializer(latest, many=False)
        return Response(serializer.data)

    @action(detail=False, methods=['POST'])
    def train(self, request):
        if request.user.id is None:
            raise PermissionDenied({
                "message": "Can only trigger training as root"
            })

        epochs = 5
        if 'epochs' in request.data:
            epochs = int(request.data['epochs'])

        train_classifier(settings.ML['TRAIN_BATCH_SIZE'],
                         settings.ML['TEST_BATCH_SIZE'],
                         num_epochs=epochs)

        return Response('Ok')


class TrainImageView(viewsets.ModelViewSet):
    queryset = TrainImage.objects.all()
    serializer_class = TrainImageSerializer

    def create(self, request, *args, **kwargs):
        req_data = request.data.copy()

        width = self.parse_int_arg(req_data.pop('width'))
        height = self.parse_int_arg(req_data.pop('height'))

        imgB64 = request.data['image']
        imgBytes = base64.decodebytes(imgB64.encode())
        img = Image.frombytes('RGBA', (width, height), imgBytes).convert('L')

        byteArr = io.BytesIO()
        img.save(byteArr, format='png')
        byteArr = byteArr.getvalue()
        req_data['image'] = base64.encodebytes(byteArr).decode('utf-8')

        serializer = self.get_serializer(data=req_data)
        serializer.is_valid(raise_exception=True)

        train_image = serializer.save()

        self.update_features(train_image, img)

        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED,
                        headers=headers)

    def parse_int_arg(self, arg):
        if type(arg) == list:
            arg = arg[0]
        if type(arg) == str:
            arg = int(arg)
        return arg

    @timeit
    def update_features(self, train_image, img):
        with torch.no_grad():
            model = ClassificationModel.get_latest().to_pytorch()

            img = mm.preprocess(img)
            img = img.repeat((3, 1, 1))
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

            features = model.features(img)
            features = features.mean([2, 3])
            byte_f = io.BytesIO()
            torch.save(features, byte_f)

            train_image.features = byte_f.getvalue()
            train_image.save()

    @action(detail=False, methods=['GET'])
    def dist(self, request):
        vals = TrainImage.objects.values('symbol') \
            .annotate(number=Count('id')) \
            .order_by('-number')
        vals = list(vals)

        labels = [MathSymbol.get(val['symbol']).name for val in vals]
        values = [val['number'] for val in vals]

        if request.GET.get('json') == '':
            response = list(map(lambda x: {"name": x[0], "number": x[1]},
                                zip(labels, values)))
            return Response(response)
        else:
            y_pos = np.arange(len(labels))

            width = 15
            height = 15
            if request.GET.get('width') is not None:
                width = int(request.GET.get('width'))
            if request.GET.get('height') is not None:
                height = int(request.GET.get('height'))

            plt.figure(figsize=(width, height), dpi=80)
            if request.GET.get('log') == '':
                plt.yscale('log')
            else:
                plt.yscale('linear')
            plt.bar(y_pos, values, align='center', alpha=0.5)
            plt.xticks(y_pos, labels)
            plt.title('Number of images per class')

            img_io = io.BytesIO()
            plt.savefig(img_io)

            return HttpResponse(img_io.getvalue(), content_type="image/png")

    @action(detail=False, methods=['GET'])
    def download(self, request):
        if request.user.id is None:
            raise PermissionDenied({
                "message": "Can only trigger training as root"
            })

        res = {
            "train_images": [],
            "symbols": []
        }

        symbols = list(MathSymbol.objects.all())
        for i, symbol in enumerate(symbols):
            res["symbols"].append({
                "id": symbol.id,
                "name": symbol.name,
                "timestamp": symbol.timestamp,
                "description": symbol.description,
                "latex": symbol.latex,
                "image": from_memoryview(symbol.image)
            })

        train_images = list(TrainImage.objects.all())
        for image in tqdm(train_images):
            res["train_images"].append({
                "symbol": image.symbol.id,
                "image": from_memoryview(image.image),
                "features": from_memoryview(image.features)
            })

        byte = io.BytesIO()
        np.save(byte, res)

        arr = byte.getvalue()

        response = HttpResponse(arr, content_type="application/octet-stream")
        response['Content-Disposition'] = 'attachment; filename="download.pth"'

        return response


def from_memoryview(data):
    if isinstance(data, memoryview):
        return data.tobytes()
    return data
