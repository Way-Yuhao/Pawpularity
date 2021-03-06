import torch
# import torchvision.models.efficientnet as effNet
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
# import tez
# import timm
# import torchvision.models as models


class PetFinderModel(nn.Module):
    def __init__(self):
        super(PetFinderModel, self).__init__()
        self.eff_net = EfficientNet.from_pretrained('efficientnet-b7')  # [m, 1000] for b1
        # self.bn1 = nn.BatchNorm1d(1000)
        self.drop_out = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=1000+12, out_features=64)
        # self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=1)

        # for p in self.eff_net.parameters():
        #     p.requires_grad = False

    def forward(self, x, meta):
        e = self.eff_net(x)  # [1, 1000] for b1
        # e_normed = self.bn1(e)
        # e_dropped = self.drop_out(e_normed)

        fr_r_meta = torch.cat((e, meta), dim=1)
        fr_r_meta_dropped = self.drop_out(fr_r_meta)

        f2 = self.fc2(fr_r_meta_dropped)  # FIXME
        # f2_r = self.rl2(f2)
        f2_r = torch.sigmoid(f2)
        # f2_rd = self.drop_out2(f2_r)

        f3 = self.fc3(f2_r)
        out = torch.sigmoid(f3) * 100

        return out


# class SwinModel(tez.Model):
#     def __init__(self, model_name):
#         super().__init__()
#         self.model = timm.create_model(model_name, pretrained=True, in_chans=3)
#         self.model.head = torch.nn.Linear(self.model.head.in_features, 128)
#         self.dropout = torch.nn.Dropout(0.1)
#         self.dense1 = torch.nn.Linear(140, 64)
#         self.dense2 = torch.nn.Linear(64, 1)
#
#     def forward(self, image, features, targets=None):
#         x = self.model(image)
#         x = self.dropout(x)
#         x = torch.cat([x, features], dim=1)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         return x, 0, {}
