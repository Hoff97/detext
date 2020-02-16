import io

import torch
from PIL import Image

from detext.server.ml.models.mobilenet import MobileNet, preprocess
from detext.server.models import TrainImage

import time


def update_train_features(torch_model, num_classes):
    with torch.no_grad():
        model = MobileNet(features=num_classes, pretrained=False)
        model.load_state_dict(torch.load(torch_model,
                              map_location=torch.device('cpu')))
        model = model.eval()

        for i, train_img in enumerate(TrainImage.objects.all()):
            if i % 10 == 0:
                print(f'Updating image {i}')
            image = Image.open(io.BytesIO(train_img.image))
            data = preprocess(image)
            img = data.repeat((3, 1, 1))
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

            features = model.features(img)
            features = features.mean([2, 3])
            byte_f = io.BytesIO()
            torch.save(features, byte_f)

            train_img.features = byte_f.getvalue()
            train_img.save()


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print(f'{method.__name__}  {(te-ts)*1000} ms')
        return result
    return timed
