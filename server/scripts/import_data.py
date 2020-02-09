import io
from pathlib import Path

import torch
from django.utils import timezone

from detext.server.models import MathSymbol, TrainImage

from tqdm import tqdm

import numpy as np


def run():
    path = 'download.pth'

    data = np.load(path, allow_pickle=True)

    symbols = data.item().get("symbols")
    train_images = data.item().get("train_images")

    sym_id = {}

    MathSymbol.objects.all().delete()
    TrainImage.objects.all().delete()

    for symbol in symbols:
        sym = MathSymbol(name=symbol.get('name'),
                         timestamp=symbol.get('timestamp'),
                         description=symbol.get('description'),
                         latex=symbol.get('latex'),
                         image=symbol.get('image'))
        sym.save()
        sym_id[symbol.get('id')] = sym

    for train_image in tqdm(train_images):
        img = TrainImage(symbol=sym_id[train_image.get('symbol')],
                         image=train_image.get('image'),
                         locked=True,
                         features=train_image.get('features'))
        img.save()