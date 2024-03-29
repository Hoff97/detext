import io
import random
import sys
from pathlib import Path

from django.db import IntegrityError, migrations, transaction
from django.utils import timezone
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets


def valid_func(x):
    return random.random() < 2

def save_instance(instance):
    try:
        with transaction.atomic():
            instance.save()
    except IntegrityError:
            instance.delete()

def load_dataset(apps, schema_editor):
    MathSymbol = apps.get_model("server", "MathSymbol")
    ClassificationModel = apps.get_model("server", "ClassificationModel")
    TrainImage = apps.get_model("server", "TrainImage")

    data_dir = 'res/datasets/math_symbols_small'
    full_dataset = datasets.ImageFolder(data_dir, is_valid_file=valid_func)

    num_imgs = len(full_dataset.imgs)
    if 'test' in sys.argv:
        num_imgs = 10

    instance_list = []
    for cl in full_dataset.classes:
        cls_instance = MathSymbol(None, cl.lower(), timezone.now())
        try:
            with transaction.atomic():
                cls_instance.save()
                instance_list.append(cls_instance)
        except IntegrityError:
            cls_instance.delete()

    for i, data in enumerate(full_dataset.imgs[:num_imgs]):
        if i%200 == 0:
            print(f"Saving image {i+1}/{len(full_dataset.imgs)}")
        img_path, ix = data
        cls_instance = instance_list[ix]
        img = Image.open(img_path)

        byteArr = io.BytesIO()
        img.save(byteArr, format='png')
        byteArr = byteArr.getvalue()
        imgInstance = TrainImage(None, symbol=cls_instance, image=byteArr, timestamp=timezone.now(), user=None, locked=False)
        save_instance(imgInstance)

    content = Path('res/mobile_cnn.onnx').read_bytes()
    modelInstance = ClassificationModel(None, model=content, timestamp=timezone.now())
    save_instance(modelInstance)

def drop_dataset(apps, schema_editor):
    pass

class Migration(migrations.Migration):

    dependencies = [
        ('server', '0002_trainimage_locked'),
    ]

    operations = [
        migrations.RunPython(load_dataset, drop_dataset, atomic=False)
    ]
