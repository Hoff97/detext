from django.db import models
from torch.utils.data import Dataset


class DBDataset(Dataset):
    def __init__(self, table: models.Model):
        self.table = table
        self.length = self.table.objects.count()
        self.entities = self.table.objects.all()

    def __getitem__(self, index):
        entity = self.entities[index]
        symbol = entity.symbol
        img_data = entity.image
        print(symbol, img_data)
