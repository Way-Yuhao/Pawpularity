import torch.nn
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset


class PetfinderDataset(Dataset):
    def __init__(self, df, image_size=224, augment=False):
        self._X = df["Id"].values
        self._y = None
        self.augment = augment
        if "Pawpularity" in df.keys():
            self._y = df["Pawpularity"].values
        if not self.augment:
            self._transform = torch.nn.Sequential(
                T.ConvertImageDtype(torch.float32),
                T.Resize([image_size, image_size]),
                T.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]))

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        image = self._transform(image)
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image
