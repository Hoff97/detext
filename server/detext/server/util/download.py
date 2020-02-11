import io

import numpy as np
from tqdm import tqdm

from detext.server.models import MathSymbol, TrainImage


def data_to_file():
    res = {
        "train_images": [],
        "symbols": []
    }

    symbols = list(MathSymbol.objects.all())
    for i, symbol in enumerate(symbols):
        res["symbols"].append({
            "id": symbol.id,
            "name": symbol.name,
            "timestamp": symbol.timestamp,
            "description": symbol.description,
            "latex": symbol.latex,
            "image": from_memoryview(symbol.image)
        })

    train_images = list(TrainImage.objects.all())
    for image in tqdm(train_images):
        res["train_images"].append({
            "symbol": image.symbol.id,
            "image": from_memoryview(image.image),
            "features": from_memoryview(image.features)
        })

    byte = io.BytesIO()
    np.save(byte, res)

    return byte


def from_memoryview(data):
    if isinstance(data, memoryview):
        return data.tobytes()
    return data
