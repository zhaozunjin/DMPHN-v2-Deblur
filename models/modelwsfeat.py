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

#
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder,self).__init__()
#         self.is_cuda = torch.cuda.is_available()
#         # Conv_1
#         self.Conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1,stride=1)
#         self.Conv1_2 =ResBlock(32, 32, kernel_size=3, padding=1,stride=1)
#         self.Conv1_3 = ResBlock(32, 32, kernel_size=3, padding=1, stride=1)
#         # Conv_2
#         self.Conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
#         self.Conv2_2 = ResBlock(64, 64, kernel_size=3, padding=1, stride=1)
#         self.Conv2_3 = ResBlock(64, 64, kernel_size=3, padding=1, stride=1)
#         # Conv_3
#         self.Conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
#         self.Conv3_2 = ResBlock(128, 128, kernel_size=3, padding=1, stride=1)
#         self.Conv3_3 = ResBlock(128, 128, kernel_size=3, padding=1, stride=1)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.xavier_uniform_(m.weight.data)
#                 init.constant_(m.bias.data, 0.1)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#
#     def forward(self, input_encoder):
#         # Conv_1
#         output_Conv1_1 = self.Conv1_1(input_encoder)
#         output_Conv1_2 = self.Conv1_2(output_Conv1_1)
#         output_Conv1_3 = self.Conv1_3(output_Conv1_1+output_Conv1_2)+output_Conv1_1+output_Conv1_2
#         # Conv_2
#         output_Conv2_1 = self.Conv2_1(output_Conv1_3)
#         output_Conv2_2 = self.Conv2_2(output_Conv2_1)
#         output_Conv2_3 = self.Conv2_3(output_Conv2_1 + output_Conv2_2) + output_Conv2_1 + output_Conv2_2
#         # Conv_3
#         output_Conv3_1 = self.Conv3_1(output_Conv2_3)
#         output_Conv3_2 = self.Conv3_2(output_Conv3_1)
#         output_Conv3_3 = self.Conv3_3(output_Conv3_1 + output_Conv3_2) + output_Conv3_1 + output_Conv3_2
#
#         return output_Conv3_3
#
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder,self).__init__()
#         # Deconv_3
#         self.Deconv3_1 = ResBlock(128, 128, kernel_size=3, padding=1, stride=1)
#         self.Deconv3_2 = ResBlock(128, 128, kernel_size=3, padding=1, stride=1)
#         self.Deconv3_3 = nn.ConvTranspose2d(128,64,kernel_size=4, stride=2, padding=1)
#         # Deconv_2
#         self.Deconv2_1 = ResBlock(64, 64, kernel_size=3, padding=1, stride=1)
#         self.Deconv2_2 = ResBlock(64, 64, kernel_size=3, padding=1, stride=1)
#         self.Deconv2_3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=1)
#         # Deconv_1
#         self.Deconv1_1 = ResBlock(32, 32, kernel_size=3, padding=1, stride=1)
#         self.Deconv1_2 = ResBlock(32, 32, kernel_size=3, padding=1, stride=1)
#         self.Deconv1_3 = nn.Conv2d(32, 3, kernel_size=3, stride=2, padding=1)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.xavier_uniform_(m.weight.data)
#                 init.constant_(m.bias.data, 0.1)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#
#     def forward(self,output_decoder):
#         # Deconv3
#         output_Deconv3_1 = self.Deconv3_1(output_decoder)
#         output_Deconv3_2 = self.Deconv3_2(output_decoder+output_Deconv3_1)+output_decoder+output_Deconv3_1
#         output_Deconv3_3 = self.Deconv3_3(output_Deconv3_2)
#         # Deconv2
#         output_Deconv2_1 = self.Deconv2_1(output_Deconv3_3)
#         output_Deconv2_2 = self.Deconv2_2(output_Deconv3_3 + output_Deconv2_1) + output_Deconv3_3 + output_Deconv2_1
#         output_Deconv2_3 = self.Deconv2_3(output_Deconv2_2)
#         # Deconv1
#         output_Deconv1_1 = self.Deconv1_1(output_Deconv2_3)
#         output_Deconv1_2 = self.Deconv1_2(output_Deconv2_3 + output_Deconv1_1) + output_Deconv2_3 + output_Deconv1_1
#         output_Deconv1_3 = self.Deconv1_3(output_Deconv1_2)
#
#         return output_Deconv1_3


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
        return H, W

    def forward(self, input_generator, feature=None, residual=None):
        H, W = self.divide_patchs(input_generator)
        # level3
        if feature is not None:
            self.feature['lv4_1'] = self.encoder_lv4_1(self.images['lv4_1']+residual['lv4_top'][:, :, :, 0:int(W / 2)])
            self.feature['lv4_2'] = self.encoder_lv4_2(self.images['lv4_2']+residual['lv4_top'][:, :, :, int(W / 2):W])
            self.feature['lv4_3'] = self.encoder_lv4_3(self.images['lv4_3']+residual['lv4_bottom'][:, :, :, 0:int(W / 2)])
            self.feature['lv4_4'] = self.encoder_lv4_4(self.images['lv4_4']+residual['lv4_bottom'][:, :, :, int(W / 2):W])
            self.feature['lv4_top'] = torch.cat((self.feature['lv4_1'], self.feature['lv4_2']), 3)
            self.feature['lv4_bottom'] = torch.cat((self.feature['lv4_3'], self.feature['lv4_4']), 3)
            # self.feature['lv4'] = torch.cat((self.feature['lv4_top'], self.feature['lv4_bottom']), 2)
            self.residual['lv4_top'] = self.decoder_lv4_1(self.feature['lv4_top']+feature['lv4_top'])
            self.residual['lv4_bottom'] = self.decoder_lv4_2(self.feature['lv4_bottom']+feature['lv4_bottom'])

        else:
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

        if feature is not None:
            self.feature['lv2_1'] = self.encoder_lv2_1(self.images['lv2_1'] + self.residual['lv4_top'] + \
                                                       residual['lv4_top']) + self.feature['lv4_top']
            self.feature['lv2_2'] = self.encoder_lv2_2(self.images['lv2_2'] + self.residual['lv4_bottom'] + \
                                                       residual['lv4_bottom']) + self.feature['lv4_bottom']
            self.feature['lv2'] = torch.cat((self.feature['lv2_1'], self.feature['lv2_2']), 2)
            self.residual['lv2'] = self.decoder_lv2_1(self.feature['lv2']+feature['lv2'])
        else:
            self.feature['lv2_1'] = self.encoder_lv2_1(self.images['lv2_1'] + self.residual['lv4_top']) + \
                                    self.feature['lv4_top']
            self.feature['lv2_2'] = self.encoder_lv2_2(self.images['lv2_2'] + self.residual['lv4_bottom']) + \
                                    self.feature['lv4_bottom']
            self.feature['lv2'] = torch.cat((self.feature['lv2_1'], self.feature['lv2_2']), 2)
            self.residual['lv2'] = self.decoder_lv2_1(self.feature['lv2'])
        # level1

        if feature is not None:
            self.feature['lv1'] = self.encoder_lv1_1(self.images['lv1_1'] + self.residual['lv2']+residual['lv2']) + self.feature['lv2']
            self.residual['lv1'] = self.decoder_lv1_1(self.feature['lv1']+feature['lv1'])
        else:
            self.feature['lv1'] = self.encoder_lv1_1(self.images['lv1_1'] + self.residual['lv2']) + self.feature['lv2']
            self.residual['lv1'] = self.decoder_lv1_1(self.feature['lv1'])
        return self.feature, self.residual, self.residual['lv1']
#
# class NestedGenerator(nn.Module):
#     def __init__(self):
#         super(NestedGenerator, self).__init__()
#         self.images = {}
#         self.feature = {}
#         self.residual = {}
#         self.Generator1 = Generator(levelnum=1)
#         self.Generator2 = Generator(levelnum=2)
#         self.Generator3 = Generator(levelnum=3)
#
#     def forward(self, x):
#         self.divide_patchs(x)
#         # level3
#         for lv in range(1, 5):
#             level = 'lv{}_{}'.format(str(3), str(lv))
#
#             self.feature[level], self.residual[level] = self.Generator3(self.images[level])
#
#         self.feature['lv3_top'] = torch.cat((self.feature['lv3_1'], self.feature['lv3_2']), 3)
#         self.feature['lv3_bot'] = torch.cat((self.feature['lv3_3'], self.feature['lv3_4']), 3)
#         self.residual['lv3_top'] = torch.cat((self.residual['lv3_1'], self.residual['lv3_2']), 3)
#         self.residual['lv3_bot'] = torch.cat((self.residual['lv3_3'], self.residual['lv3_4']), 3)
#
#         # level2
#         self.feature['lv2_1'], self.residual['lv2_1'] = self.Generator2(self.images['lv2_1']+self.residual['lv3_top'], self.feature['lv3_top'])
#         self.feature['lv2_1'] += self.feature['lv3_top']
#         self.feature['lv2_2'], self.residual['lv2_2'] = self.Generator2(self.images['lv2_2']+self.residual['lv3_bot'], self.feature['lv3_bot'])
#         self.feature['lv2_2'] += self.feature['lv3_bot']
#
#         self.feature['lv2'] = torch.cat((self.feature['lv2_1'], self.feature['lv2_2']), 2)
#         self.residual['lv2'] = torch.cat((self.residual['lv2_1'], self.residual['lv2_2']), 2)
#
#         # level1
#         self.feature['lv1'], self.residual['lv1'] = self.Generator1(self.images['lv1_1']+self.residual['lv2'], self.feature['lv2'])
#         return self.feature['lv1'], self.residual['lv1']


class StackShareNet(nn.Module):
    def __init__(self):
        super(StackShareNet, self).__init__()
        self.basicnet = WSDMPHN()

    def forward(self, x):
        feature, residual,x1 = self.basicnet(x)
        feature, residual,x2 = self.basicnet(x1, feature, residual)
        _, _, x3 = self.basicnet(x2, feature, residual)
        return x1, x2, x3

# class Discriminator(nn.Module):
#     def __init__(self, opt):
#         super(Discriminator, self).__init__()
#         N = opt.nfc
#         self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
#         self.body = nn.Sequential()
#         for i in range(opt.num_layer - 2):
#             N = int(opt.nfc / pow(2, (i + 1)))
#             block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
#             self.body.add_module('block%d' % (i + 1), block)
#         self.tail = nn.Conv2d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.head = Encoder()
        self.tail = nn.Sequential(nn.Conv2d(128, 1, kernel_size=3, padding=1),
                                  nn.Sigmoid())

    def forward(self, input_D):
        feature = self.encoder(input_D)
        output = self.tail(feature)
        return output



