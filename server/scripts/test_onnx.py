import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision import datasets

import scripts.models.mobilenet as mm

from torch.utils.data import DataLoader


def run():
    ort_session = ort.InferenceSession('res/mobile_cnn.onnx')

    data_dir = 'res/test'
    full_dataset = datasets.ImageFolder(data_dir, mm.preprocess)
    dataloader = DataLoader(full_dataset, batch_size=1, shuffle=True,
                            num_workers=4)

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.numpy()

        outputs = ort_session.run(None, {'input.1': inputs})
        _, preds = torch.max(torch.from_numpy(np.array(outputs)), 2)
        print(preds, full_dataset.classes[labels], outputs)
