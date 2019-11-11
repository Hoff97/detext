# Generated by Django 2.2.6 on 2019-11-09 09:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('server', '0003_auto_20191029_2316'),
    ]

    operations = [
        migrations.AddField(
            model_name='mathsymbol',
            name='description',
            field=models.CharField(default='', max_length=1000),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='mathsymbol',
            name='image',
            field=models.BinaryField(blank=True, editable=True),
        ),
        migrations.AddField(
            model_name='mathsymbol',
            name='latex',
            field=models.CharField(default='', max_length=200),
            preserve_default=False,
        ),
    ]