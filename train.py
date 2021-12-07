import cv2
import io
import torch
import torch.nn as nn
import torch.optim as optim
import sys
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

from data_loader import PetFinderDataset
from model import PetFinderModel
import torch.multiprocessing


"""Global Parameters"""
CUDA_DEVICE = "cuda:7"
dataset_path = "/mnt/data1/yl241/datasets/Pawpularity/"
network_weight_path = "./weight/"
model_name = None
version = None
# num_workers_train = 4  # FIXME
# batch_size = 4
num_workers_train = 24  # FIXME
batch_size = 24
n_splits = 5  # FIXME


"""Hyper Parameters"""
init_lr = 1e-5
epoch = 500


def print_params():
    print("######## Basics ##################")
    print("version: {}".format(version))
    print("Training on {}".format(CUDA_DEVICE))


def print_params_2(num_train_batches, num_dev_batches):
    print("# of training minibatches = {} | dev = {}".format(num_train_batches, num_dev_batches))


def load_data(df):
    data_loader = torch.utils.data.DataLoader(PetFinderDataset(df), batch_size=batch_size,
                                              num_workers=num_workers_train, drop_last=True)
    return data_loader


def load_network_weights(net, path):
    print("loading pre-trained weights from {}".format(path))
    net.load_state_dict(torch.load(path))


def save_network_weights(net, ep=None):
    filename = network_weight_path + "{}{}_epoch_{}.pth".format(model_name, version, ep)
    torch.save(net.state_dict(), filename)
    print("network weights saved to ", filename)
    return


def compute_loss(output, target):
    mse_criterion = nn.MSELoss()
    mse_loss = mse_criterion(output.squeeze(1), target)
    return mse_loss


def train_dev(net, tb, load_weights=False, pre_trained_params_path=None):
    print_params()
    net.to(CUDA_DEVICE)
    net.train()

    if load_weights:
        load_network_weights(net, pre_trained_params_path)
    df = pd.read_csv(p.join(dataset_path, "train.csv"))
    df["Id"] = df["Id"].apply(lambda x: os.path.join(dataset_path, "train", x + ".jpg"))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)  # FIXME
    # fold, (train_idx, val_idx) = enumerate(skf.split(df["Id"], df["Pawpularity"]))
    enum = enumerate(skf.split(df["Id"], df["Pawpularity"]))
    fold, (train_idx, val_idx) = list(enum)[0]
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    train_loader = load_data(train_df)
    dev_loader = load_data(val_df)
    train_num_mini_batches = len(train_loader)
    dev_num_mini_batches = len(dev_loader)

    # optimizer = optim.Adam(net.parameters(), lr=init_lr)
    optimizer = optim.AdamW(net.parameters(), lr=init_lr, weight_decay=1.)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=.8)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=.96)

    print_params_2(train_num_mini_batches, dev_num_mini_batches)
    running_train_loss, running_dev_loss = 0.0, 0.0  # per epoch
    train_output, dev_output = None, None
    train_input, dev_input = None, None
    label = None
    for ep in range(epoch):
        print("Epoch", ep)
        train_iter, dev_iter = iter(train_loader), iter(dev_loader)
        # TRAIN
        for _ in tqdm(range(train_num_mini_batches)):
            net.train()
            train_input, meta, label = train_iter.next()
            train_input, meta, label = train_input.to(CUDA_DEVICE), meta.to(CUDA_DEVICE), label.to(CUDA_DEVICE)
            train_output = net(train_input, meta)  # [m, c, h, w]
            train_loss = compute_loss(train_output, label)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
            pass
        with torch.no_grad():
            # net.eval()
            for _ in range(dev_num_mini_batches):
                dev_input, meta, label = dev_iter.next()
                dev_input, meta, label = dev_input.to(CUDA_DEVICE), meta.to(CUDA_DEVICE), label.to(CUDA_DEVICE)
                dev_output = net(dev_input, meta)
                dev_loss = compute_loss(dev_output, label)
                running_dev_loss += dev_loss.item()
        scheduler.step()
        # record loss values after each epoch
        cur_train_loss = running_train_loss / train_num_mini_batches
        cur_dev_loss = running_dev_loss / dev_num_mini_batches
        print("train loss = {:.4} | val loss = {:.4}".format(cur_train_loss, cur_dev_loss))
        tb.add_scalar('loss/lr', scheduler._last_lr[0], ep)
        tb.add_scalar('loss/train', cur_train_loss, ep)
        tb.add_scalar('loss/dev', cur_dev_loss, ep)
        tb.add_histogram('distribution of dev output', dev_output, ep)

        if ep % 5 == 4:
            save_network_weights(net, ep="{}".format(ep))  # FIXME
            # input_img_grid = torchvision.utils.make_grid(train_input)
            # tb.add_image("{}/inputs".format("train"), input_img_grid, global_step=ep)
            pass

        running_train_loss, running_dev_loss = 0.0, 0.0

    print("finished training")
    tb.add_histogram('distribution of dev labels', label, 0)
    save_network_weights(net, ep="{}_FINAL".format(epoch))


def predict(net, load_weights=True, pre_trained_params_path=None):
    if load_weights:
        load_network_weights(net, pre_trained_params_path)
    else:
        raise Exception("ERROR: need to load network weight")
    df = pd.read_csv(p.join(dataset_path, "test.csv"))
    df["Id"] = df["Id"].apply(lambda x: os.path.join(dataset_path, "train", x + ".jpg"))
    test_loader = torch.utils.data.DataLoader(PetFinderDataset(df), batch_size=batch_size,
                                              num_workers=8, drop_last=False)
    test_num_mini_batches = len(test_loader)
    predictions = np.array([])
    net.eval()
    for _ in tqdm(range(test_num_mini_batches)):
        test_input, meta, label = test_loader.next()
        test_output = net(test_input, meta)
        predictions = np.append(predictions, test_output)

    df["Pawpularity"] = predictions
    df = df[["Id", "Pawpularity"]]
    df.to_csv("./submission.csv", index=False)


def train_simple(net, tb, load_weights=False, pre_trained_params_path=None):
    print_params()
    net.to(CUDA_DEVICE)
    net.train()

    if load_weights:
        load_network_weights(net, pre_trained_params_path)
    df = pd.read_csv(p.join(dataset_path, "train_20.csv"))
    df["Id"] = df["Id"].apply(lambda x: os.path.join(dataset_path, "train", x + ".jpg"))
    train_loader = load_data(df)
    train_num_mini_batches = len(train_loader)

    optimizer = optim.Adam(net.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=.96)
    running_train_loss = 0.0  # per epoch

    for ep in range(epoch):
        print("Epoch", ep)
        train_iter = iter(train_loader)
        # TRAIN
        for _ in tqdm(range(train_num_mini_batches)):
            net.train()
            train_input, meta, label = train_iter.next()
            train_input, meta, label = train_input.to(CUDA_DEVICE), meta.to(CUDA_DEVICE), label.to(CUDA_DEVICE)
            train_output = net(train_input, meta)  # [m, c, h, w]
            train_loss = compute_loss(train_output, label)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
        scheduler.step()
        # record loss values after each epoch
        cur_train_loss = running_train_loss / train_num_mini_batches
        print("train loss = {:.4} ".format(cur_train_loss))
        tb.add_scalar('loss/train', cur_train_loss, ep)
        tb.add_scalar('loss/lr', scheduler._last_lr[0], ep)
        if ep % 10 == 9:
            # save_network_weights(net, ep="{}".format(ep))  # FIXME
            input_img_grid = torchvision.utils.make_grid(train_input)
            tb.add_image("{}/inputs".format("train"), input_img_grid, global_step=ep)
            tb.add_histogram('distribution of output', train_output, ep)
            pass
        running_train_loss, running_dev_loss = 0.0, 0.0
    print("finished training")
    tb.add_histogram('distribution of labels', label, 0)
    save_network_weights(net, ep="{}_FINAL".format(epoch))


def main():
    global model_name, version
    # sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
    # sys.path.append('../input/tez-lib')
    model_name = "CNN"
    version = "-v0.7.0"
    # param_to_load = "./weight/CNN{}_epoch_{}.pth".format(version, "100_FINAL")
    param_to_load = None
    tb = SummaryWriter('./runs/' + model_name + version)

    net = PetFinderModel()  # TODO
    # net = SwinModel(model_name="swin_large_patch4_window12_384")
    # net.load(f"../input/paw-models/model_f0.bin", device=CUDA_DEVICE, weights_only=True)
    train_dev(net, tb, load_weights=False, pre_trained_params_path=param_to_load)
    tb.close()


if __name__ == "__main__":
    main()
