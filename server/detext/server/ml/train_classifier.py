import gc
import io
import random

import torch
import torch.nn as nn
from django.utils import timezone

from detext.server.ml.models.linear import LinearModel
from detext.server.ml.training.train import Solver
from detext.server.ml.util.util import eval_model
from detext.server.models import ClassificationModel
from scripts.train import setup_db_dl


def valid_func(x):
    return random.random() < 2


def get_data(item):
    features = torch.load(io.BytesIO(item.features),
                          map_location=torch.device('cpu'))\
                              .detach().reshape((-1))
    return features


def get_label(item):
    return item.symbol.name


def get_class_name(item):
    return item.name


def train_classifier(train_batch_size=16,
                     test_batch_size=4,
                     device="cpu",
                     num_epochs=5):

    criterion = nn.CrossEntropyLoss()

    dataloaders, full_dataset = setup_db_dl(train_batch_size, test_batch_size,
                                            get_data)

    print(dataloaders)

    old_model = ClassificationModel.get_latest().to_pytorch()

    n_features = full_dataset.get_input_shape()[0]
    n_classes = full_dataset.num_classes

    model = LinearModel(n_features, n_classes)

    model = model.to(device)

    solver = Solver(criterion, dataloaders, model)
    model, accuracy = solver.train(device=device,
                                   num_epochs=num_epochs,
                                   step_size=2)

    eval_model(model, dataloaders["test"], device, len(full_dataset.classes))

    model = model.to('cpu')
    old_model.set_classifier(model.classifier)
    old_model = old_model.eval()

    del dataloaders
    del full_dataset
    gc.collect()

    byteArr = io.BytesIO()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(old_model, dummy_input, byteArr)

    torchByteArr = io.BytesIO()
    torch.save(old_model.state_dict(), torchByteArr)

    model_entity = ClassificationModel(None, model=byteArr.getvalue(),
                                       timestamp=timezone.now(),
                                       pytorch=torchByteArr.getvalue(),
                                       accuracy=accuracy)
    model_entity.save()
