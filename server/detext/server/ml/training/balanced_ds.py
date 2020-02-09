import torch

from detext.server.ml.training.dataloader import DBDataset
from detext.server.ml.training.remap_ds import RemapDS


class BalancedDS(RemapDS):
    def __init__(self, ds: DBDataset, min_count=1):
        self.ds = ds

        self.class_counts = ds.class_counts
        max_count = max(torch.max(self.class_counts),min_count)

        multiples = [int(num.item()) for num in torch.floor(max_count / self.class_counts)]
        new_cls_counts = self.class_counts * multiples
        print(new_cls_counts)

        indice_maps = []
        for i in range(len(ds)):
            label = ds.get_lbl(i)

            num = multiples[label]
            indice_maps.extend([i]*num)

        super().__init__(ds, indice_maps)

        self.classes = ds.classes
