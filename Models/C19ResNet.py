import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, \
    wide_resnet50_2, wide_resnet101_2

resnet_models = {'resnet18': resnet18,
                 'resnet34': resnet34,
                 'resnet50': resnet50,
                 'resnet101': resnet101,
                 'resnet152': resnet152,
                 'resnext50_32x4d': resnext50_32x4d,
                 'resnext101_32x8d': resnext101_32x8d,
                 'wide_resnet50_2': wide_resnet50_2,
                 'wide_resnet101_2': wide_resnet101_2}


class C19ResNet(nn.Module):
    def __init__(self, pretrained=True, model='resnet18', RGB_input=True):
        super().__init__()
        self.backbone = resnet_models[model](pretrained=pretrained)
        self.fc = nn.Linear(in_features=self.backbone.fc.in_features, out_features=1)
        self.RGB_input = RGB_input
        if not RGB_input:
            self.conv = nn.Conv2d(1, 3, 1)
            self.conv.weight.data.fill_(1)
            self.conv.bias.data.fill_(0)

    def forward(self, x):
        if not self.RGB_input:
            x = self.conv(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        x = x.view(x.size(0), self.backbone.fc.in_features)
        x = self.fc(x)

        return x