import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision import datasets

import models.mobilenet as mm

from torch.utils.data import DataLoader

# Load the ONNX model
model = onnx.load("res/mobile_cnn.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

ort_session = ort.InferenceSession('res/mobile_cnn.onnx')

data_dir = 'res/test'
full_dataset = datasets.ImageFolder(data_dir, mm.preprocess)
dataloader = DataLoader(full_dataset, batch_size=1, shuffle=True, num_workers=4)

for i, data in enumerate(dataloader):
    inputs, labels = data
    inputs = inputs.numpy()

    outputs = ort_session.run(None, {'input.1': inputs})
    _, preds = torch.max(torch.from_numpy(np.array(outputs)), 2)
    print(preds, full_dataset.classes[labels], outputs)
