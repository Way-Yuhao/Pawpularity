import torchvision.models.efficientnet as effNet
from efficientnet_pytorch import EfficientNet
import torch.nn as nn



class EfficientNetModel(nn.Module):
    def __init__(self):
        model = EfficientNet.from_name('efficientnet-b1')
        super(model, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        img_input = nn.Input(shape=(224, 224, 3))
        meta_input = nn.Input(shape=(32,))

        X = model(img_input)
        X = tfl.BatchNormalization()(X)

        con = tfl.concatenate([X, meta_input])

        X = tfl.Dense(64, activation='relu')(con)
        X = tfl.Dense(64, activation='relu')(X)

        X = tfl.Dropout(0.3)(X)

        out = tfl.Dense(1)(X)

        model = keras.Model(inputs=[img_input, meta_input], outputs=out)

        return model