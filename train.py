import cv2
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import os.path as p
from tqdm import tqdm
from matplotlib import pyplot as plt
from data_loader import PetfinderDataset
import pandas as pd

"""Global Parameters"""
dataset_path = "/mnt/data1/yl241/datasets/Pawpularity/"
network_weight_path = "./weight/"
model_name = None
version = None
num_workers_train = 8
batch_size = 8


"""Hyper Parameters"""
init_lr = 5e-1
epoch = 2000


def load_data(df):
    data_loader = torch.utils.data.DataLoader(PetfinderDataset(df))
    return data_loader


def main():
    df = pd.read_csv(p.join(dataset_path, "train.csv"))
    df["Id"] = df["Id"].apply(lambda x: os.path.join(dataset_path, "train", x + ".jpg"))
    data_loader = load_data(df)
    images, labels = iter(data_loader).next()
    print(images)
    print(labels)


if __name__ == "__main__":
    main()
