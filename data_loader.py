import numpy as np
import torch.nn
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import albumentations
import cv2


class PetFinderDataset(Dataset):
    def __init__(self, df, image_size=384, augment=False):
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
        self._transform = T.Compose([
            lambda x: x / 255,
            T.Resize([image_size, image_size]),
            T.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path).type(torch.float32)
        image = self._transform(image)
        image = self.data_augmentation(image)

        meta = self.meta[idx]
        if self._y is not None:
            label = self._y[idx]
            return image, meta, label
        return image, meta

    def data_augmentation(self, input_):
        """
        applies a sequence of data augmentations
        :param input_: CMOS input
        :param spad: SPAD input
        :param target: ground truth
        :return: augmented inputs and ground truth
        """
        if self.augment is False:
            pass
        else:
            input_ = self.random_horizontal_flip(input_)
        return input_

    def random_horizontal_flip(self, input_, p=.5):
        """
        applies a random horizontal flip
        :param input_: CMOS input
        :param spad: SPAD input
        :param target: ground truth
        :param p: probability of applying the flip
        :return: flipped or original images
        """
        x = np.random.rand()
        if x < p:
            input_ = torch.flip(input_, (2,))
        return input_