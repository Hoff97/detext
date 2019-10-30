import torch
import torch.nn.functional as F
from django.db import models
from torch.utils.data import Dataset


class DBDataset(Dataset):
    def __init__(self, data_table: models.Model, class_table: models.Model, get_data, get_label, get_class_name, filter = None):
        self.data_table = data_table

        if filter is not None:
            self.entities = [row for row in self.data_table.objects.all() if filter(row)]
        else:
            self.entities = list(self.data_table.objects.all())

        self.len = len(self.entities)

        self.classes = [get_class_name(cls_ent) for cls_ent in class_table.objects.all().order_by('timestamp')]
        self.class_to_ix = {
            get_class_name(cls_ent): ix for ix, cls_ent in enumerate(class_table.objects.all().order_by('timestamp'))
        }
        self.num_classes = len(self.classes)
        self.get_data = get_data
        self.get_label = get_label

    def __getitem__(self, index):
        entity = self.entities[index]
        data = self.get_data(entity)
        label = self.get_label(entity)
        label_tensor = self.class_to_ix[label]
        return data, label_tensor

    def __len__(self):
        return self.len
