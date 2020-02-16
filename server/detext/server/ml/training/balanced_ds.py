import torch

from detext.server.ml.training.dataloader import DBDataset
from detext.server.ml.training.remap_ds import RemapDS

import math


class BalancedDS(RemapDS):
    def __init__(self, ds: DBDataset, min_count=1):
        self.ds = ds

        self.class_counts = ds.class_counts
        max_count = torch.max(self.class_counts)
        max_count = max(math.ceil(math.sqrt(max_count)), min_count)

        cls_count = torch.clamp(self.class_counts, 1)

        multiples = [int(num.item())
                     for i, num in enumerate(torch.ceil(max_count / cls_count))]

        indice_maps = []
        for i in range(len(ds)):
            label = ds.get_lbl(i)

            num = multiples[label]
            indice_maps.extend([i]*num)

        super().__init__(ds, indice_maps)

        self.classes = ds.classes
