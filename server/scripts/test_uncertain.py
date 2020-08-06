from detext.server.ml.models.mobilenet import MobileNet, preprocess

from PIL import Image

import torch
import torch.nn.functional as F

from detext.server.models import MathSymbol

from torchvision import datasets

from torch.utils.data import DataLoader


def get_class_name(item):
    return item.name


def run():
    class_table = MathSymbol
    classes = [
        get_class_name(cls_ent)
        for cls_ent in class_table.objects.all().order_by('timestamp')
    ]
    class_to_ix = {
        get_class_name(cls_ent): ix for ix, cls_ent in
        enumerate(class_table.objects.all().order_by('timestamp'))
    }
    print(class_to_ix)

    file_name = "test_augment.pth"

    model = MobileNet.from_file(file_name, test_time_dropout=False,
                                estimate_variane=True)
    model.eval().cuda()

    iters = 200

    full_dataset = datasets.ImageFolder("res/certain", preprocess)
    dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False,
                            num_workers=4)

    for i, data in enumerate(dataloader):
        preds = torch.zeros((iters, 63))
        softmaxs = torch.zeros((iters, 63))
        maxs = torch.zeros(iters)
        variances = None

        inputs, labels = data

        print(labels)

        inputs = inputs.cuda()

        for j in range(iters):
            res = model(inputs)
            pred = res[0]
            var = res[1]
            pred, var = pred.detach().cpu(), var.detach().cpu()
            preds[j] = pred[0]
            maxs[j] = pred.argmax()
            softmaxs[j] = F.softmax(pred[0])
            variances = var[0]

        stds = softmaxs.std(dim=0)
        m = int(maxs[0].item())

        print(m)
        print(stds[m]/0.001)
        print(variances[m]/0.001
