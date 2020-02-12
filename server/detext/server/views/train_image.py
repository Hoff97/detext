import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import torch.__config__
from django.db.models import Count
from django.http import HttpResponse
from PIL import Image
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied
from rest_framework.response import Response

import detext.server.ml.models.mobilenet as mm
from detext.server.models import ClassificationModel, MathSymbol, TrainImage
from detext.server.serializers import TrainImageSerializer
from detext.server.util.transfer import data_to_file
from detext.server.util.util import timeit


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

        byte = data_to_file()

        arr = byte.getvalue()

        response = HttpResponse(arr, content_type="application/octet-stream")
        response['Content-Disposition'] = 'attachment; filename="download.pth"'

        return response
