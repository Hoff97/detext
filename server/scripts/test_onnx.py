import copy
import random
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets

import models.cnn as cm
import models.mobilenet as mm
from models.mobilenet import MobileNet
from training.train import train_model

# Load the ONNX model
model = onnx.load("res/mobile_cnn.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

ort_session = ort.InferenceSession('res/mobile_cnn.onnx')

data_dir = 'res/test'
full_dataset = datasets.ImageFolder(data_dir, mm.preprocess)
dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=1, shuffle=True, num_workers=4)

for i, data in enumerate(dataloader):
    inputs, labels = data
    inputs = inputs.numpy()

    outputs = ort_session.run(None, {'input.1': inputs})
    _, preds = torch.max(torch.from_numpy(np.array(outputs)), 2)
    print(preds, full_dataset.classes[labels], outputs)
