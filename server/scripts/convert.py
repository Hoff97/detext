import onnx
from detext.server.models import ClassificationModel

from detext.server.ml.models.mobilenet import MobileNet
import torch

from django.utils import timezone

import io


def run():
    file_name = "test_augment.pth"

    model = MobileNet.from_file(file_name, test_time_dropout=False,
                                estimate_variane=True)

    byteArr = model.to_onnx()

    torchByteArr = io.BytesIO()
    torch.save(model.state_dict(), torchByteArr)

    model_entity = ClassificationModel(None, model=byteArr.getvalue(),
                                       timestamp=timezone.now(),
                                       pytorch=torchByteArr.getvalue(),
                                       accuracy=0.99)
    model_entity.save()
