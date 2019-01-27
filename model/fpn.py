import torch
import math
import torch.nn as nn



class ConvBasic(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, k=7, same=True):
        super(ConvBasic, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, k, padding= (k - 1) // 2 if same else 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias= False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(residual + out)

class FPN(nn.Module):
    def __init__(self, block, layers, in_channels=3, upsample_mode = True):
        super(FPN, self).__init__()
        self.inplanes = 64
        self.in_channels = in_channels
        self.conv1 = ConvBasic(self.in_channels, 64, 7)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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

        self.upLayer = nn.ConvTranspose2d(256, 256, kernel_size=1) if not upsample_mode else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upLayer2 = nn.ConvTranspose2d(256, 256, kernel_size=1) if not upsample_mode else nn.Upsample(scale_factor=2, mode='bilinear')
        self.upLayer3 = nn.ConvTranspose2d(256, 256, kernel_size=1) if not upsample_mode else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
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

    # def upsample_add(self, x, y, mode='bilinear'):
    #     _, _, H, W = y.size()
    #     if mode == 'bilinear':
    #         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    #     else:
    #         self.up = nn.ConvTranspose2d()
    #     return F.upsample(x, ) if mode == 'bilinear' else nn.ConvTranspose2d(s)

    def forward(self, x):
        x = self.conv1(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(x)
        c4 = self.layer3(x)
        c5 = self.layer4(x)


        p5 = self.toplayer(c5)
        p4 = self.upLayer1(p5) + self.latlayer1(c4)
        p3 = self.upLayer2(p4) + self.latlayer2(c3)
        p2 = self.upLayer3(p3) + self.latlayer2(c2)
        return [p2, p3, p4, p5]


if __name__ == '__main__':
    a = torch.randn(1, 3, 30, 30)
    model = FPN(BottleNeck, [3, 4, 6, 3])
    # for i in model.named_modules():
    #     print(i)
    # print(model)
    b = model(a)
    print(b)
