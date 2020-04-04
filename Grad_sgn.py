import torch.nn as nn
import torch


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

        self.body = nn.Sequential(*layers)

    def forward(self, x, sigma):
        noise = 1 / (sigma ** 2)
        out = self.body(x)
        out = x - out - noise * x

        return out

class PixelUnShuffel(nn.Module):
    def __init__(self):
        super(PixelUnShuffel, self).__init__()

    def forward(self, x):
        N, C, H, W = x.shape

        x1 = x[:, :, 0:H:2, 0:W:2]
        x2 = x[:, :, 0:H:2, 1:W:2]
        x3 = x[:, :, 1:H:2, 0:W:2]
        x4 = x[:, :, 1:H:2, 1:W:2]

        out = torch.cat((x1, x2, x3, x4), dim=1)

        return out


class MiniSGN(nn.Module):
    def __init__(self, feats=64):
        super(MiniSGN, self).__init__()
        self.PixelUnShuffle = PixelUnShuffel()
        self.PixelShuffle = nn.PixelShuffle(upscale_factor=2)

        self.up_layers = ConvBlock(in_channels=feats*4, feats=feats*4, out_channels=feats*4)
        self.down_layers = ConvBlock(in_channels=feats*2, feats=feats*2, out_channels=feats)

        self.tau1 = nn.Conv2d(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.tau2 = nn.Conv2d(in_channels=feats, out_channels=feats, kernel_size=1, stride=1, padding=0, bias=False)



    def forward(self, x, sigma):
        noise = 1 / (sigma ** 2)

        out = self.PixelUnShuffle(x)
        out = self.up_layers(out)
        out = self.PixelShuffle(out)

        out = self.down_layers(torch.cat((out, x), dim=1))
        out = x - self.tau1(out) - self.tau2(noise * x)

        return out


class GRAD(nn.Module):
    def __init__(self, feats=64, basic_conv=1, tail_conv=2):
        super(GRAD, self).__init__()

        layers_head = [ConvBlock(in_channels=3, feats=feats, out_channels=feats)]
        self.head = nn.Sequential(*layers_head)

        self.basic_block1 = MiniSGN(feats=feats)
        self.basic_block2 = MiniSGN(feats=feats)
        self.basic_block3 = MiniSGN(feats=feats)
        self.basic_block4 = MiniSGN(feats=feats)
        self.basic_block5 = MiniSGN(feats=feats)
        self.basic_block6 = MiniSGN(feats=feats)
        self.basic_block7 = MiniSGN(feats=feats)
        self.basic_block8 = MiniSGN(feats=feats)

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

    def forward(self, x, sigma):
        noise = 1/ (sigma ** 2 )

        x0 = self.head(x)

        out = self.basic_block1(x0, sigma)
        out = out + self.tau1(noise * x0)

        out = self.basic_block2(out, sigma)
        out = out + self.tau2(noise * x0)

        out = self.basic_block3(out, sigma)
        out = out + self.tau3(noise * x0)

        out = self.basic_block4(out, sigma)
        out = out + self.tau4(noise * x0)

        out = self.basic_block5(out, sigma)
        out = out + self.tau5(noise * x0)

        out = self.basic_block6(out, sigma)
        out = out + self.tau6(noise * x0)

        out = self.basic_block7(out, sigma)
        out = out + self.tau7(noise * x0)

        out = self.basic_block8(out, sigma)
        out = out + self.tau8(noise * x0)

        out = self.tail(out)

        return out
