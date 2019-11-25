import io
import random
from pathlib import Path

from django.db import IntegrityError, migrations, transaction
from django.utils import timezone
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets


def save_instance(instance):
    try:
        with transaction.atomic():
            instance.save()
    except IntegrityError:
            instance.delete()

def load_dataset(apps, schema_editor):
    ClassificationModel = apps.get_model("server", "ClassificationModel")

    content = Path('res/mobile_cnn.onnx').read_bytes()
    pytorch = Path('res/mobile_cnn.pth').read_bytes()
    modelInstance = ClassificationModel(None, model=content, pytorch=pytorch, timestamp=timezone.now())
    save_instance(modelInstance)

def drop_dataset(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0008_classificationmodel_pytorch'),
    ]

    operations = [
        migrations.RunPython(load_dataset, drop_dataset, atomic=False)
    ]
