from torch.utils.data import Dataset


class RemapDS(Dataset):
    def __init__(self, ds: Dataset, indice_maps):
        self.ds = ds

        self.indice_maps = indice_maps

        self.len = len(indice_maps)

    def __getitem__(self, index):
        return self.ds[self.indice_maps[index]]

    def __len__(self):
        return self.len
