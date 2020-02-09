import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image
from albumentations import (Compose, ElasticTransform, ShiftScaleRotate)
from torchvision import transforms


def eval_model(model, test_dl, device, n_classes):
    conf_matrix = torch.zeros(n_classes, n_classes)

    for i, data in enumerate(test_dl):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for p, t in zip(preds, labels):
                conf_matrix[p, t] += 1

    print(conf_matrix)

    df_cm = pd.DataFrame(conf_matrix.numpy(), index=[i for i in range(n_classes)],
                         columns=[i for i in range(n_classes)])
    plt.figure(figsize=(16, 12))
    sn.heatmap(df_cm, annot=True)

    plt.savefig('conf.png')


preprocess = transforms.ToTensor()


def augment_image(image):
    p = 0.0
    if image.size[0] < 224:
        p = 1.0

    scale = ShiftScaleRotate(shift_limit=0.0, scale_limit=(-0.5, -0.2),
                             rotate_limit=0, interpolation=cv2.INTER_CUBIC,
                             border_mode=cv2.BORDER_CONSTANT,
                             value=(255, 255, 255), p=p)
    elastic = ElasticTransform(alpha=0.5, sigma=20, alpha_affine=20, border_mode=cv2.BORDER_CONSTANT,
                               value=(255, 255, 255), interpolation=cv2.INTER_CUBIC, p=1.0)
    aug = Compose([scale, elastic])

    image = image.resize((224, 224), resample=Image.LANCZOS)

    img = np.array(image.convert('RGB'))
    img = aug(image=img)['image']

    img = Image.fromarray(img)

    return preprocess(img)
