import torch
import torchvision.models.efficientnet as effNet
from efficientnet_pytorch import EfficientNet
import torch.nn as nn


class PetFinderModel(nn.Module):
    def __init__(self):
        super(PetFinderModel, self).__init__()
        self.eff_net = EfficientNet.from_pretrained('efficientnet-b5')  # [m, 1000] for b1
        self.bn1 = nn.BatchNorm1d(num_features=1000)
        self.fc1 = nn.Linear(in_features=1000, out_features=64)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.rl2 = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x, meta):
        e = self.eff_net(x)  # [1, 1000] for b1
        e_normed = self.bn1(e)
        f1 = self.fc1(e_normed)
        f1_r = self.rl1(f1)

        # fr_r_meta = torch.cat((f1_r, meta), dim=1)

        f2 = self.fc2(f1_r)  # FIXME
        f2_r = self.rl2(f2)
        f2_o = self.drop_out(f2_r)
        out = self.fc3(f2_o)
        return out