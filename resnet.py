import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#from IPython import embed

def default_conv(in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1, args=None):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=stride, bias=bias, groups=groups)
    return m


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, conv3x3=default_conv, args=None):
        modules = [conv3x3(in_channels, out_channels, kernel_size, stride=stride, bias=bias, args=args),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True)]
        super(BasicBlock, self).__init__(*modules)



def make_model(args, parent=False):
    return ResNet(args[0])




class ResBlock(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, stride=1, conv3x3=default_conv, downsample=None, args=None):
        super(ResBlock, self).__init__()

        self.stride = stride
        m = [conv3x3(in_channels, planes, kernel_size, stride=stride, bias=False, args=args),
             nn.BatchNorm2d(planes),
             nn.ReLU(inplace=True),
             conv3x3(planes, planes, kernel_size, bias=False, args=args),
             nn.BatchNorm2d(planes)]

        self.body = nn.Sequential(*m)
        self.downsample = downsample
        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.body(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, stride=1, conv3x3=default_conv,
                 conv1x1=default_conv, downsample=None, args=None):
        super(BottleNeck, self).__init__()

        expansion = 4
        m = [conv1x1(in_channels, planes, 1, bias=False),
             nn.BatchNorm2d(planes),
             nn.ReLU(inplace=True),
             conv3x3(planes, planes, kernel_size, stride=stride, bias=False, args=args),
             nn.BatchNorm2d(planes),
             nn.ReLU(inplace=True),
             conv1x1(planes, expansion * planes, 1, bias=False),
             nn.BatchNorm2d(expansion * planes)]

        self.body = nn.Sequential(*m)
        self.downsample = downsample
        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.body(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out


class DownSampleA(nn.Module):
    def __init__(self):
        super(DownSampleA, self).__init__()

    def forward(self, x):
        # identity shortcut with 'zero padding' in the channel dimension
        c = x.size(1) // 2
        pool = F.avg_pool2d(x, 2)
        out = F.pad(pool, [0, 0, 0, 0, c, c], 'constant', 0)

        return out


class DownSampleC(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, conv1x1=default_conv):
        m = [conv1x1(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels)]
        super(DownSampleC, self).__init__(*m)


class ResNet(nn.Module):
    def __init__(self, args, conv3x3=default_conv, conv1x1=default_conv):
        super(ResNet, self).__init__()

        n_classes = args.n_class
        kernel_size = 3
        if args.depth == 50:
            self.expansion = 4
            self.block = BottleNeck
            self.n_blocks = (args.depth - 2) // 9
        elif args.depth <= 56:
            self.expansion = 1
            self.block = ResBlock
            self.n_blocks = (args.depth - 2) // 6
        else:
            self.expansion = 4
            self.block = BottleNeck
            self.n_blocks = (args.depth - 2) // 9
        self.in_channels = 16
        self.downsample_type = args.downsample_type
        bias = False

        kwargs = {'conv3x3': conv3x3,
                  'conv1x1': conv1x1,
                  'args': args}
        stride = 1 
        m = [BasicBlock(3, 16, kernel_size=kernel_size, stride=stride, bias=bias, conv3x3=conv3x3, args=args),
             self.make_layer(self.n_blocks, 16, kernel_size, **kwargs),
             self.make_layer(self.n_blocks, 32, kernel_size, stride=2, **kwargs),
             self.make_layer(self.n_blocks, 64, kernel_size, stride=2, **kwargs),
             nn.AvgPool2d(8)]
        fc = nn.Linear(64 * self.expansion, n_classes)

        self.features = nn.Sequential(*m)
        self.classifier = fc

    def make_layer(self, blocks, planes, kernel_size, stride=1, conv3x3=default_conv, conv1x1=default_conv,
                   args=None):
        out_channels = planes * self.expansion
        if stride != 1 or self.in_channels != out_channels:
            if self.downsample_type == 'A':
                downsample = DownSampleA()
            elif self.downsample_type == 'C':
                downsample = DownSampleC(self.in_channels, out_channels, stride=stride, conv1x1=conv1x1)
        else:
            downsample = None
        kwargs = {'conv3x3': conv3x3,
                  'args': args}
        if self.block == BottleNeck:
            kwargs['conv1x1'] = conv1x1

        m = [self.block(self.in_channels, planes, kernel_size, stride=stride, downsample=downsample, **kwargs)]
        self.in_channels = out_channels
        for _ in range(blocks - 1):
            m.append(self.block(self.in_channels, planes, kernel_size, **kwargs))

        return nn.Sequential(*m)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.squeeze())
        return x

