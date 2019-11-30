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

from detext.server.util.train import train_classifier


def run():
    train_classifier(device='cuda')