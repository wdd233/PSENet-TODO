import torch
import math
import torch.nn as nn

class FPN(nn.Module):
    def __init__(self, block, layers, in_channels):
        super(FPN, self).__init__()
        self.inplanes = 64
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padidng=1)
        self.layer1 = self._make_layers(block, 64, layers[0])
        self.layer2 = self._make_layers(block, 128, layers[1])
        self.layer3 = self._make_layers(block, 256, layers[2])
        self.layer4 = self._make_layers(block, 512, layers[3])

        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1)

        self.smooth1= nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2= nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3= nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[9] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.dta.zero_()
    def _make_layers(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes !=block.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, block.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(block.expansion * planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def upsample_add(self, mode=''):


