import torch
import torchvision.models.efficientnet as effNet
from efficientnet_pytorch import EfficientNet
import torch.nn as nn


class PetFinderModel(nn.Module):
    def __init__(self):
        super(PetFinderModel, self).__init__()
        self.eff_net = EfficientNet.from_pretrained('efficientnet-b1')  # [m, 1000] for b1
        self.bn1 = nn.BatchNorm2d(num_features=1000)
        self.dense_net1 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        # TODO concatenate with feature

    def forward(self, x):
        e = self.eff_net(x)  # [1, 1000] for b1
        e_normed = self.bn1(e)

        return x


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

        return model