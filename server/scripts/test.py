import random

import torch
from torchvision import datasets

import detext.server.ml.models.mobilenet as mm

from torch.utils.data import DataLoader

from detext.server.models import ClassificationModel, MathSymbol


def valid_func(x):
    return random.random() < 0.1


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    names = [ms.name for ms in MathSymbol.objects.all().order_by('timestamp')]
    print(names)

    data_dir = 'res/test'
    full_dataset = datasets.ImageFolder(data_dir, mm.preprocess)
    dataloader = DataLoader(full_dataset, batch_size=1, shuffle=True,
                            num_workers=4)

    model = ClassificationModel.get_latest().to_pytorch()

    model.eval()
    model = model.to(device)

    correct = 0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        actual_name, pred_name = full_dataset.classes[labels], names[preds]
        if actual_name == pred_name:
            correct += 1
        else:
            print(actual_name, pred_name)

    print(f'Correct: {correct}/{len(dataloader)}')
