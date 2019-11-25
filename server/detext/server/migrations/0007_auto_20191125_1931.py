import io

from django.db import migrations
from PIL import Image


def update_images(apps, schema_editor):
    TrainImage = apps.get_model("server", "TrainImage")

    train_images = list(TrainImage.objects.all())
    for train_image in train_images:
        img = Image.open(io.BytesIO(train_image.image)).convert('L')

        byteArr = io.BytesIO()
        img.save(byteArr, format='png')
        byteArr = byteArr.getvalue()
        train_image.image = byteArr
        train_image.save()

def drop_dataset(apps, schema_editor):
    pass

class Migration(migrations.Migration):

    dependencies = [
        ('server', '0006_auto_20191120_2225'),
    ]

    operations = [
        migrations.RunPython(update_images, drop_dataset, atomic=False)
    ]
