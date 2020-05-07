from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import os

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))


class ResBlock(nn.Sequential):
    def __init__(self,in_channel, out_channel, kernel_size, padding, stride):
        super(ResBlock,self).__init__()
        self.add_module('Conv1', nn.Conv2d(in_channel, out_channel, kernel_size, padding, stride))
        self.add_module('Relu', nn.ReLU(inplace=True))
        self.add_module('Conv2', nn.Conv2d(in_channel, out_channel, kernel_size, padding, stride))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        # Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        # Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 修改Conv1的连接方式
        output_layer1 = self.layer1(x)
        output_layer2 = self.layer2(output_layer1)
        output_layer3 = self.layer3(output_layer2 + output_layer1) + output_layer2 + output_layer1
        output_layer4 = self.layer4(
            output_layer3 + output_layer2 + output_layer1) + output_layer3 + output_layer2 + output_layer1

        # 修改Conv2的连接方式
        output_layer5 = self.layer5(output_layer4)
        output_layer6 = self.layer6(output_layer5)
        output_layer7 = self.layer7(output_layer6 + output_layer5) + output_layer6 + output_layer5
        output_layer8 = self.layer8(
            output_layer7 + output_layer6 + output_layer5) + output_layer7 + output_layer6 + output_layer5

        # 修改Conv3的连接方式
        output_layer9 = self.layer9(output_layer8)
        output_layer10 = self.layer10(output_layer9)
        output_layer11 = self.layer11(output_layer10 + output_layer9) + output_layer10 + output_layer9
        output_layer12 = self.layer12(
            output_layer11 + output_layer10 + output_layer9) + output_layer11 + output_layer10 + output_layer9

        return output_layer12


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # 修改Deconv3的连接方式
        output_layer13 = self.layer13(x)
        output_layer14 = self.layer14(output_layer13 + x) + output_layer13 + x
        # output_layer15 = self.layer15(output_layer14+output_layer13 + x) + output_layer14+output_layer13 + x
        output_layer16 = self.layer16(output_layer14)

        # 修改Deconv2的连接方式
        output_layer17 = self.layer17(output_layer16)
        output_layer18 = self.layer18(output_layer17 + output_layer16) + output_layer17 + output_layer16
        # output_layer19 = self.layer19(output_layer18+output_layer17 + output_layer16) + output_layer18+output_layer17 + output_layer16
        output_layer20 = self.layer20(output_layer18)

        # 修改Conv1的连接方式
        output_layer21 = self.layer21(output_layer20)
        output_layer22 = self.layer22(output_layer21 + output_layer20) + output_layer21 + output_layer20
        # output_layer23 = self.layer22(output_layer22+output_layer21 + output_layer20) + output_layer22+output_layer21 + output_layer20
        output_layer24 = self.layer24(output_layer22)
        return output_layer24


class WSDMPHN(nn.Module):
    def __init__(self):
        super(WSDMPHN, self).__init__()
        self.images = {}
        self.feature = {}
        self.residual = {}
        self.encoder_lv4_1 = Encoder()
        self.encoder_lv4_2 = Encoder()
        self.encoder_lv4_3 = Encoder()
        self.encoder_lv4_4 = Encoder()
        self.decoder_lv4_1 = Decoder()
        self.decoder_lv4_2 = Decoder()
        # self.encoder_lv4 = Encoder()
        # self.decoder_lv4 = Decoder()

        self.encoder_lv2_1 = Encoder()
        self.encoder_lv2_2 = Encoder()
        self.decoder_lv2_1 = Decoder()
        # self.encoder_lv2 = Encoder()
        # self.decoder_lv2 = Decoder()

        self.encoder_lv1_1 = Encoder()
        self.decoder_lv1_1 = Decoder()

    def divide_patchs(self, images):
        H = images.size(2)
        W = images.size(3)
        self.images['lv1_1'] = images
        self.images['lv2_1'] = self.images['lv1_1'][:, :, 0:int(H / 2), :]
        self.images['lv2_2'] = self.images['lv1_1'][:, :, int(H / 2):H, :]
        self.images['lv4_1'] = self.images['lv2_1'][:, :, :, 0:int(W / 2)]
        self.images['lv4_2'] = self.images['lv2_1'][:, :, :, int(W / 2):W]
        self.images['lv4_3'] = self.images['lv2_2'][:, :, :, 0:int(W / 2)]
        self.images['lv4_4'] = self.images['lv2_2'][:, :, :, int(W / 2):W]


    def forward(self, input_generator):
        self.divide_patchs(input_generator)
        # level3
        self.feature['lv4_1'] = self.encoder_lv4_1(self.images['lv4_1'])
        self.feature['lv4_2'] = self.encoder_lv4_2(self.images['lv4_2'])
        self.feature['lv4_3'] = self.encoder_lv4_3(self.images['lv4_3'])
        self.feature['lv4_4'] = self.encoder_lv4_4(self.images['lv4_4'])
        self.feature['lv4_top'] = torch.cat((self.feature['lv4_1'], self.feature['lv4_2']), 3)
        self.feature['lv4_bottom'] = torch.cat((self.feature['lv4_3'], self.feature['lv4_4']), 3)
        # self.feature['lv4'] = torch.cat((self.feature['lv4_top'], self.feature['lv4_bottom']), 2)
        self.residual['lv4_top'] = self.decoder_lv4_1(self.feature['lv4_top'])
        self.residual['lv4_bottom'] = self.decoder_lv4_2(self.feature['lv4_bottom'])
        # level2
        self.feature['lv2_1'] = self.encoder_lv2_1(self.images['lv2_1']+self.residual['lv4_top'])+self.feature['lv4_top']
        self.feature['lv2_2'] = self.encoder_lv2_2(self.images['lv2_2']+self.residual['lv4_bottom'])+self.feature['lv4_bottom']
        self.feature['lv2'] = torch.cat((self.feature['lv2_1'], self.feature['lv2_2']), 2)
        self.residual['lv2'] = self.decoder_lv2_1(self.feature['lv2'])
        # level1
        self.feature['lv1'] = self.encoder_lv1_1(self.images['lv1_1'] + self.residual['lv2'])+self.feature['lv2']
        self.residual['lv1'] = self.decoder_lv1_1(self.feature['lv1'])
        return self.residual['lv1']


class StackShareNet(nn.Module):
    def __init__(self):
        super(StackShareNet, self).__init__()
        self.basicnet = WSDMPHN()

    def forward(self, x):
        x1 = self.basicnet(x)
        x2 = self.basicnet(x1)
        x3 = self.basicnet(x2)
        return x1, x2, x3




