import copy
import io
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from django.utils import timezone
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import datasets

import scripts.models.mobilenet as mm
from detext.server.models import TrainImage

def update_train_features(torch_model, num_classes):
    with torch.no_grad():
        model = mm.MobileNet(features=num_classes, pretrained=False)
        model.load_state_dict(torch.load(torch_model, map_location=torch.device('cpu')))
        model = model.eval()

        for i, train_img in enumerate(TrainImage.objects.all()):
            if i%10 == 0:
                print(f'Updating image {i}')
            image = Image.open(io.BytesIO(train_img.image))
            data = mm.preprocess(image)
            img = data.repeat((3,1,1))
            img = img.reshape((1,img.shape[0], img.shape[1], img.shape[2]))

            features = model.features(img)
            features = features.mean([2, 3])
            byte_f = io.BytesIO()
            torch.save(features, byte_f)

            train_img.features = byte_f.getvalue()
            train_img.save()