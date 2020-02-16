import base64

from django.conf import settings
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied
from rest_framework.response import Response

from detext.server.ml.train_classifier import train_classifier
from detext.server.models import ClassificationModel
from detext.server.serializers import ClassificationModelSerializer


class ClassificationModelView(viewsets.ViewSet):
    queryset = ClassificationModel.objects.all()
    serializer_class = ClassificationModelSerializer

    @action(detail=False)
    def latest(self, request):
        latest = ClassificationModel.objects.all()\
            .order_by('-timestamp').first()

        if request.GET.get('timestamp') is not None:
            ts = parse_datetime(request.GET.get('timestamp'))
            if ts == latest.timestamp:
                return Response(status=status.HTTP_304_NOT_MODIFIED)

        serializer = ClassificationModelSerializer(latest, many=False)
        return Response(serializer.data)

    def create(self, request):
        if request.user.id is None:
            raise PermissionDenied({
                "message": "Can only create model as root"
            })

        pytorch = request.data['pytorch']
        pytorch = base64.decodebytes(pytorch.encode())

        onnx = request.data['onnx']
        onnx = base64.decodebytes(onnx.encode())

        model_instance = ClassificationModel(None, model=onnx,
                                             pytorch=pytorch,
                                             timestamp=timezone.now(),
                                             accuracy=0.99)
        model_instance.save()

        return Response('Ok')

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
