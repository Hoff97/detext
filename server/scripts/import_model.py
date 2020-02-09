from pathlib import Path

from django.utils import timezone

from detext.server.models import ClassificationModel

from detext.server.ml.models.mobilenet import MobileNet


def run():
    path = 'test_augment_2.pth'

    pytorch = Path(path).read_bytes()

    model = MobileNet.from_file(path)
    model.eval()

    byte_arr = model.to_onnx()

    model_instance = ClassificationModel(None, model=byte_arr.getvalue(),
                                         pytorch=pytorch,
                                         timestamp=timezone.now(),
                                         accuracy=0.99)
    model_instance.save()
