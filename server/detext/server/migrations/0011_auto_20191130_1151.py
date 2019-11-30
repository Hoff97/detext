# Generated by Django 2.2.6 on 2019-11-30 11:51

from django.db import migrations

import io
import random
from pathlib import Path

from django.db import IntegrityError, migrations, transaction
from django.utils import timezone

from detext.server.util.util import update_train_features

def update_features(apps, schema_editor):
    ClassificationModel = apps.get_model("server", "ClassificationModel")
    MathSymbol = apps.get_model("server", "MathSymbol")

    current_model = ClassificationModel.objects.all().order_by('-timestamp').first()
    old_classes = MathSymbol.objects.all().filter(timestamp__lte=current_model.timestamp)

    update_train_features(io.BytesIO(current_model.pytorch), len(old_classes))

def drop_dataset(apps, schema_editor):
    pass

class Migration(migrations.Migration):

    dependencies = [
        ('server', '0010_trainimage_features'),
    ]

    operations = [
        migrations.RunPython(update_features, drop_dataset, atomic=False)
    ]