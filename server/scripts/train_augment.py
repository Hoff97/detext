import io

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, random_split

from detext.server.ml.models.mobilenet import MobileNet
from detext.server.ml.training.balanced_ds import BalancedDS
from detext.server.ml.training.dataloader import DBDataset
from detext.server.ml.training.train import Solver
from detext.server.ml.util.util import eval_model, Augmenter
from detext.server.models import MathSymbol, TrainImage


def open_image(item):
    return Image.open(io.BytesIO(item.image))


def get_data(augmenter):
    def load(item):
        image = open_image(item)

        return augmenter.augment_image(image)
    return load


def get_label(item):
    return item.symbol.name


def get_class_name(item):
    return item.name


def run(num_epochs=5, device="cuda"):
    criterion = nn.CrossEntropyLoss()

    augmenter = Augmenter(approximate=True)

    full_dataset = DBDataset(TrainImage, MathSymbol, get_data(augmenter),
                             get_label, get_class_name)

    full_dataset = BalancedDS(full_dataset)

    test_train_split = 0.9
    train_size = int(test_train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset,
                                               [train_size, test_size])

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=32, num_workers=1),
        "test":  DataLoader(test_dataset, batch_size=16, shuffle=True,
                            num_workers=1)
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "test": len(test_dataset)
    }

    print(dataloaders)
    print(dataset_sizes)

    model = MobileNet(features=len(full_dataset.classes), pretrained=True)
    model.freeze()
    model = model.to(device)

    def unfreeze(x, y):
        model.unfreeze()
        augmenter.approximate = False

    solver = Solver(criterion, dataloaders, model, cb=unfreeze)
    model, accuracy = solver.train(device=device,
                                   num_epochs=num_epochs)

    model = model.to('cpu')
    torch.save(model.state_dict(), "test_augment.pth")

    model = model.to(device)
    eval_model(model, dataloaders["test"], device, len(full_dataset.classes))
