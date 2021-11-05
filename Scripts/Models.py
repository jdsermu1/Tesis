from collections import OrderedDict

from torch import nn
from torchvision import models
import torch


class ModelGenerator:
    def __init__(self, device, num_classes):
        super()
        self.device = device
        self.num_classes = num_classes


    def resnet(self, version, pretrained, freeze, dense, ordinal = False):
        model = getattr(models, version)(pretrained=pretrained)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
        fc = []
        for i, layer in enumerate(dense):
            fc.append((f"linear{i}", nn.Linear(model.fc.in_features if i == 0 else dense[i - 1], layer)))
            fc.append((f"relu{i}", nn.ReLU()))
        fc.append((f'linear{len(dense)}', nn.Linear(dense[-1], self.num_classes)))
        if not ordinal:
            fc.append((f"logSoftmax", nn.LogSoftmax(dim=1)))
        else:
            fc.append((f"Sigmoid", nn.Sigmoid()))
        model.fc = nn.Sequential(OrderedDict(fc))
        model = model.to(self.device)
        return model, version + str(freeze) + '-'.join(map(str, dense))


    def li2019(self, version: int):
        assert 1 <= version <= 2
        if version == 1:
            blocks = [(32, 5, 1.0/1.8, -1), (64, 3, 1.0/1.8, -1), (96, 3, 1.0/1.8, -1), (128, 3, 1.0/1.8, -1),
                      (160, 3, 1.0/1.8, -1), (192, 3, 1.0/1.8, -1), (224, 3, 1.0/1.8, -1),
                      (256, 3, 1.0/1.8, 32.0 / 352), (288, 2, 1.0/1.8, 32.0 / 384), (320, 3, 1, 64.0 / 416),
                      (356, 1, 1, 64.0 / 448)]
            linear = 1424
        else:
            blocks = [(32, 5, 1.0/1.5, -1), (64, 3, 1.0/1.5, -1), (96, 3, 1.0/1.5, -1), (128, 3, 1.0/1.5, -1),
                      (160, 3, 1.0/1.5, -1), (192, 3, 1.0/1.5, -1), (224, 3, 1.0/1.5, -1), (256, 3, 1.0/1.5, -1),
                      (288, 3, 1.0/1.5, -1), (320, 3, 1.0/1.5, -1), (352, 3, 1.0/1.5, 32.0/352),
                      (384, 2, 1.0/1.5, 32.0/384), (416, 2, 1, 64.0/416), (448, 1, 1, 64.0/448)]
            linear = 4032
        model = ModelLi2019(blocks, self.num_classes, linear)
        model = model.to(self.device)
        return model, f"li2019-{str(version)}"


    def ghosh2017(self):
        model = ModelGhosh2017(self.num_classes)
        model = model.to(self.device)
        return model, "ghosh2017"



class BaseBlockLi2019(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fractional, dropout):
        super(BaseBlockLi2019, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.333)
        self.fractional = fractional
        self.p = dropout
        if self.fractional != 1:
            self.fractionalMaxPooling = nn.FractionalMaxPool2d(kernel_size, output_ratio=self.fractional)
        if self.p > 0.0:
            self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.leakyRelu(x)
        if self.fractional != 1:
            x = self.fractionalMaxPooling(x)
        if self.p > 0.0:
            x = self.dropout(x)
        return x


class ModelLi2019(nn.Module):
    def __init__(self, blocks, num_classes, linear):
        super(ModelLi2019, self).__init__()
        array = []
        for i, block in enumerate(blocks):
            in_channels = 3 if i == 0 else blocks[i-1][0]
            array.append((f"baseBlock{i}", BaseBlockLi2019(in_channels, block[0], block[1],
                                                           block[2], block[3])))
        self.features = nn.Sequential(OrderedDict(array))
        self.linear = nn.Linear(linear, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.logSoftmax(x)
        return x


class MaxoutGhosh2017(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m



class ModelGhosh2017(nn.Module):

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)

    def __init__(self, num_classes):
        super(ModelGhosh2017, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            MaxoutGhosh2017(16384, 512, 2),
            nn.Dropout(),
            MaxoutGhosh2017(512, 512, 2),
            nn.Dropout(),
            nn.Linear(512, 10),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.linear = nn.Linear(10, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        nn.init.xavier_normal_(self.linear.weight)
        self.features.apply(self.weights_init)
        self.fc.apply(self.weights_init)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        x = self.linear(x)
        x = self.logSoftmax(x)
        return x