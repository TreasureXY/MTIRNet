import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, feats, out_channels):
        layers = []
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=feats, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=feats, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True))


        super(ConvBlock, self).__init__(*layers)


class BasicBlock(nn.Module):
    def __init__(self, feats=64, conv_block=1):
        super(BasicBlock, self).__init__()

        layers = [ConvBlock(in_channels=feats, feats=feats, out_channels=feats) for i in range(conv_block)]
        layers.append(nn.Conv2d(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, padding=0, bias=False))

        self.body = nn.Sequential(*layers)

        self.tau = nn.Conv2d(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.body(x)
        out = x - out - self.tau(x)

        return out


class GRAD(nn.Module):
    def __init__(self, feats=64, basic_conv=1, tail_conv=2):
        super(GRAD, self).__init__()

        layers_head = [ConvBlock(in_channels=4, feats=feats, out_channels=feats)]
        self.head = nn.Sequential(*layers_head)

        self.basic_block1 = BasicBlock(feats=feats, conv_block=basic_conv)
        self.basic_block2 = BasicBlock(feats=feats, conv_block=basic_conv)
        self.basic_block3 = BasicBlock(feats=feats, conv_block=basic_conv)
        self.basic_block4 = BasicBlock(feats=feats, conv_block=basic_conv)
        self.basic_block5 = BasicBlock(feats=feats, conv_block=basic_conv)
        self.basic_block6 = BasicBlock(feats=feats, conv_block=basic_conv)
        self.basic_block7 = BasicBlock(feats=feats, conv_block=basic_conv)
        self.basic_block8 = BasicBlock(feats=feats, conv_block=basic_conv)

        self.tau1 = nn.Conv2d(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.tau2 = nn.Conv2d(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.tau3 = nn.Conv2d(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.tau4 = nn.Conv2d(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.tau5 = nn.Conv2d(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.tau6 = nn.Conv2d(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.tau7 = nn.Conv2d(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.tau8 = nn.Conv2d(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, padding=0, bias=False)

        layers_tail = [ConvBlock(in_channels=feats, feats=feats, out_channels=feats) for i in range(tail_conv)]

        layers_tail.append(nn.Conv2d(in_channels=feats, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True))
        self.tail = nn.Sequential(*layers_tail)

    def forward(self, x):

        x0 = self.head(x)

        out = self.basic_block1(x0)
        out = out + self.tau1(x0)

        out = self.basic_block2(out)
        out = out + self.tau2(x0)

        out = self.basic_block3(out)
        out = out + self.tau3(x0)

        out = self.basic_block4(out)
        out = out + self.tau4(x0)

        out = self.basic_block5(out)
        out = out + self.tau5(x0)

        out = self.basic_block6(out)
        out = out + self.tau6(x0)

        out = self.basic_block7(out)
        out = out + self.tau7(x0)

        out = self.basic_block8(out)
        out = out + self.tau8(x0)

        out = self.tail(out)

        return out
