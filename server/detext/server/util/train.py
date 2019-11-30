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

from detext.server.models import MathSymbol, TrainImage
from scripts.training.dataloader import DBDataset
from scripts.training.train import train_model
from scripts.models.linear import LinearModel

from detext.server.models import ClassificationModel, MathSymbol, TrainImage

import scripts.models.mobilenet as mm
import gc


def valid_func(x):
    return random.random() < 2

def get_data(item):
    features = torch.load(io.BytesIO(item.features), map_location=torch.device('cpu')).detach().reshape((-1))
    return features

def get_label(item):
    return item.symbol.name

def get_class_name(item):
    return item.name

def train_classifier(train_batch_size=16, test_batch_size=4, transfer_learn = False, device = "cpu"):
    criterion = nn.CrossEntropyLoss()

    full_dataset = DBDataset(TrainImage, MathSymbol, get_data, get_label, get_class_name, filter=valid_func)
    test_train_split = 0.9
    train_size = int(test_train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    weights = torch.zeros(len(train_dataset))
    for i, data in enumerate(train_dataset):
        weights[i] = 1. / full_dataset.class_counts[data[1]]

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloaders = {
        "train": torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, num_workers=1, sampler=sampler),
        "test":  torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "test": len(test_dataset)
    }

    print(dataloaders)
    print(dataset_sizes)

    latest_model = ClassificationModel.objects.all().order_by('-timestamp').first()
    state_dict = torch.load(io.BytesIO(latest_model.pytorch), map_location=torch.device('cpu'))
    old_model = mm.MobileNet(features=state_dict['mobilenet.classifier.1.bias'].shape[0], pretrained=False)
    old_model.load_state_dict(state_dict)

    n_features = full_dataset.get_input_shape()[0]
    n_classes = full_dataset.num_classes

    model = LinearModel(n_features, n_classes)
    if transfer_learn:

        weight = torch.tensor(old_model.classifier[1].weight.detach())
        bias = torch.tensor(old_model.classifier[1].bias.detach())
        w = torch.zeros((n_classes, n_features))
        b = torch.zeros(n_classes)
        nn.init.normal_(w, 0, 0.01)
        w[:len(old_classes),:] = weight
        b[:len(old_classes)] = bias
        model.classifier[1].weight.data = w
        model.classifier[1].bias.data = b

    model = model.to(device)

    model, accuracy = train_model(model, criterion, dataloaders, dataset_sizes, device, num_epochs = 5, step_size=2)

    model = model.to('cpu')
    old_model.set_classifier(model.classifier)
    old_model = old_model.eval()

    del dataloaders
    del dataset_sizes
    del sampler
    del weights
    del train_dataset
    del test_dataset
    del full_dataset
    gc.collect()

    byteArr = io.BytesIO()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(old_model, dummy_input, byteArr)

    torchByteArr = io.BytesIO()
    torch.save(old_model.state_dict(), torchByteArr)

    model_entity = ClassificationModel(None, model=byteArr.getvalue(), timestamp=timezone.now(), pytorch=torchByteArr.getvalue(), accuracy=accuracy)
    model_entity.save()