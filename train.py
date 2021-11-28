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
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data_loader import PetfinderDataset

"""Global Parameters"""
CUDA_DEVICE = "cuda:6"
dataset_path = "/mnt/data1/yl241/datasets/Pawpularity/"
network_weight_path = "./weight/"
model_name = None
version = None
num_workers_train = 8
batch_size = 8
n_splits = 5


"""Hyper Parameters"""
init_lr = 5e-1
epoch = 2000


def print_params():
    print("######## Basics ##################")
    print("version: {}".format(version))
    print("Training on {}".format(CUDA_DEVICE))


def load_data(df):
    data_loader = torch.utils.data.DataLoader(PetfinderDataset(df))
    return data_loader


def load_network_weights(net, pre_trained_params_path):
    raise NotImplementedError


def save_network_weights(net, ep=None):
    filename = network_weight_path + "{}{}_epoch_{}.pth".format(model_name, version, ep)
    torch.save(net.state_dict(), filename)
    print("network weights saved to ", filename)
    return


def compute_loss(output, target):
    mse_criterion = nn.MSELoss()
    mse_loss = mse_criterion(output, target)
    return mse_loss


def train(net, tb, load_weights=False, pre_trained_params_path=None):
    net.to(CUDA_DEVICE)
    net.train()

    if load_weights:
        load_network_weights(net, pre_trained_params_path)
    df = pd.read_csv(p.join(dataset_path, "train.csv"))
    df["Id"] = df["Id"].apply(lambda x: os.path.join(dataset_path, "train", x + ".jpg"))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)  # FIXME
    train_idx, val_idx = enumerate(skf.split(df["Id"], df["Pawpularity"]))
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    train_loader = load_data(train_df)
    dev_loader = load_data(val_df)
    train_num_mini_batches = len(train_loader)
    dev_num_mini_batches = len(dev_loader)

    optimizer = optim.Adam(net.parameters(), lr=init_lr)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=.8)

    running_train_loss, running_dev_loss = 0.0, 0.0  # per epoch
    for ep in range(epoch):
        print("Epoch", ep)
        train_iter, dev_iter = iter(train_loader), iter(dev_loader)
        # TRAIN
        for _ in tqdm(range(train_num_mini_batches)):
            input_, label = train_iter.next()
            input_, label = input_.to(CUDA_DEVICE), label.to(CUDA_DEVICE)
            output = net(input_)  # [m, c, h, w]
            train_loss = compute_loss(output, label)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
        with torch.no_grad():
            for _ in range(dev_num_mini_batches):
                input_, label = dev_iter.next()
                input_, label = input_.to(CUDA_DEVICE), label.to(CUDA_DEVICE)
                output = net(input_)
                dev_loss = compute_loss(output, label)
                running_dev_loss += dev_loss.item()

        # record loss values after each epoch
        cur_train_loss = running_train_loss / train_num_mini_batches
        cur_dev_loss = running_dev_loss / dev_num_mini_batches
        print("train loss = {:.4} | val loss = {:.4}".format(cur_train_loss, cur_dev_loss))
        tb.add_scalar('loss/train', cur_train_loss, ep)
        tb.add_scalar('loss/dev', cur_dev_loss, ep)

        if ep % 10 == 0:
            pass  # TODO

        running_train_loss, running_dev_loss = 0.0, 0.0
        # scheduler.step()

        print("finished training")
        save_network_weights(net, ep="{}_FINAL".format(epoch))


def train_k_fold_valid(net, df):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=.8)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df["Id"], df["Pawpularity"])):
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        train_loader = load_data(train_df)
        val_loader = load_data(val_df)

        num_mini_batches = len(train_loader)


def main():
    global model_name, version
    model_name = "CNN"
    version = "-v0.0.0"
    param_to_load = None
    tb = SummaryWriter('./runs/' + model_name + version)

    net = None  # TODO
    train(net, tb, load_weights=False, pre_trained_params_path=param_to_load)
    tb.close()


if __name__ == "__main__":
    main()
