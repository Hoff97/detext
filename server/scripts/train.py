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

import scripts.models.cnn as cm
import scripts.models.mobilenet as mm
from detext.server.models import ClassificationModel, MathSymbol, TrainImage
from scripts.training.dataloader import DBDataset
from scripts.training.train import train_model

# TODO: Change dataloader so images and classes come from DB, save best model to DB

def valid_func(x):
    return random.random() < 0.01

def get_data(item):
    image = Image.open(io.BytesIO(item.image))
    data = mm.preprocess(image)
    return data.repeat((3,1,1))

def get_label(item):
    return item.symbol.name

def get_class_name(item):
    return item.name

def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    full_dataset = DBDataset(TrainImage, MathSymbol, get_data, get_label, get_class_name, filter=valid_func)
    test_train_split = 0.9
    train_size = int(test_train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    dataloaders = {
        "train": torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1),
        "test":  torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=1)
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "test": len(test_dataset)
    }

    print(dataloaders)
    print(dataset_sizes)

    model = mm.MobileNet(features=len(full_dataset.classes), pretrained=False)
    #model = cm.CNNNet(len(full_dataset.classes))
    model = model.to(device)

    model = train_model(model, criterion, dataloaders, dataset_sizes, device, num_epochs = 5)

    byteArr = io.BytesIO()
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, byteArr)
    print(byteArr)
    model_entity = ClassificationModel(None, model=byteArr.getvalue(), timestamp=timezone.now())
    model_entity.save()

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    def valid_func(x):
        return random.random() < 2

    data_dir = 'res/datasets/math_symbols_small'
    full_dataset = datasets.ImageFolder(data_dir, mm.preprocess, is_valid_file=valid_func)
    test_train_split = 0.9
    train_size = int(test_train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


    dataloaders = {
        "train": torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4),
        "test":  torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "test": len(test_dataset)
    }

    print(dataloaders)
    print(dataset_sizes)

    print(full_dataset.classes)

    model = mm.MobileNet(features=len(full_dataset.classes), pretrained=False)
    #model = cm.CNNNet(len(full_dataset.classes))
    model = model.to(device)

    model = train_model(model, criterion, dataloaders, dataset_sizes, device, num_epochs = 5)

    model = model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    print(model(dummy_input))
    torch.onnx.export(model, dummy_input, "res/mobile_cnn.onnx")
    torch.save(model, "res/mobile_cnn.pth")
