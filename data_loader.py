import numpy as np
import torch.nn
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset


class PetFinderDataset(Dataset):
    def __init__(self, df, image_size=224, augment=False):
        self._X = df["Id"].values
        self._y = None
        self.augment = augment
        if "Pawpularity" in df.keys():
            self._y = torch.tensor(df["Pawpularity"].values, dtype=torch.float32)

        # handeling metadata
        b = df["Subject Focus"].values.reshape(-1, 1)
        c = df["Eyes"].values.reshape(-1, 1)
        d = df["Face"].values.reshape(-1, 1)
        e = df["Near"].values.reshape(-1, 1)
        f = df["Action"].values.reshape(-1, 1)
        g = df["Accessory"].values.reshape(-1, 1)
        h = df["Group"].values.reshape(-1, 1)
        i = df["Collage"].values.reshape(-1, 1)
        j = df["Human"].values.reshape(-1, 1)
        k = df["Occlusion"].values.reshape(-1, 1)
        l = df["Info"].values.reshape(-1, 1)
        m = df["Blur"].values.reshape(-1, 1)
        self.meta = torch.tensor(np.hstack((b, c, d, e, f, g, h, i, j, k, l, m)), dtype=torch.float32)
        if not self.augment:
            self._transform = T.Compose([
                # T.ToTensor(),
                # T.ConvertImageDtype(torch.float32),
                # lambda x: x / 255,
                T.Resize([image_size, image_size]),
                # T.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                        std=[0.229, 0.224, 0.225])])
                T.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                       std=[0.5, 0.5, 0.5])])
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path).type(torch.float32)
        image = image / 255

        my_img = image.type(torch.float32) / 255
        my_img = (my_img - .5) / .5  # FIXME delete this

        image = self._transform(image)

        meta = self.meta[idx]
        if self._y is not None:
            label = self._y[idx]
            return image, meta, label
        return image, meta
