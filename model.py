import torch
import torchvision.models.efficientnet as effNet
from efficientnet_pytorch import EfficientNet
import torch.nn as nn


class PetFinderModel(nn.Module):
    def __init__(self):
        super(PetFinderModel, self).__init__()
        self.eff_net = EfficientNet.from_pretrained('efficientnet-b1')  # [m, 1000] for b1
        self.bn1 = nn.BatchNorm1d(num_features=1000)
        self.fc1 = nn.Linear(in_features=1000, out_features=64)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.rl2 = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        e = self.eff_net(x)  # [1, 1000] for b1
        e_normed = self.bn1(e)
        f1 = self.fc1(e_normed)
        f1_r = self.rl1(f1)
        f2 = self.fc2(f1_r)
        f2_r = self.rl2(f2)
        f2_o = self.drop_out(f2_r)
        out = self.fc3(f2_o)
        return out

        # self.flatten = nn.Flatten()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(28 * 28, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 10)
        # )
        # img_input = nn.Input(shape=(224, 224, 3))
        # meta_input = nn.Input(shape=(32,))

        # X = model(img_input)
        # X = tfl.BatchNormalization()(X)
        #
        # con = tfl.concatenate([X, meta_input])
        #
        # X = tfl.Dense(64, activation='relu')(con)
        # X = tfl.Dense(64, activation='relu')(X)
        #
        # X = tfl.Dropout(0.3)(X)
        #
        # out = tfl.Dense(1)(X)
        #
        # model = keras.Model(inputs=[img_input, meta_input], outputs=out)