import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        #self.layer4 = nn.Sequential(
        #    nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #    ) 
        self.ca1 = ChannelAttention(32)
        self.sa1= SpatialAttention()
        #self.relu1 = nn.ReLU(inplace=True)       
        #Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #self.layer8 = nn.Sequential(
        #    nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #    ) 
        self.ca2 = ChannelAttention(64)
        self.sa2 = SpatialAttention()
        #self.relu2 = nn.ReLU(inplace=True)
        #Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        #self.layer12 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1)
        #    ) 
        self.ca3 = ChannelAttention(128)
        self.sa3 = SpatialAttention()
        #self.relu3 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        #Conv1
        # x = self.layer1(x)
        # x = self.layer2(x) + x
        # x = self.layer3(x) + x

        # 修改Conv1的连接方式
        output_layer1 = self.layer1(x)
        output_layer2 = self.layer2(output_layer1)
        output_layer3 = self.layer3(output_layer2 + output_layer1) + output_layer2 + output_layer1

        out = self.ca1(output_layer3) * output_layer3
        out = self.sa1(out) * out
        #output_layer3 = self.relu1(output_layer3)
        output_layer3 =out+output_layer3
        

        #Conv2
        # x = self.layer5(x)
        # x = self.layer6(x) + x
        # x = self.layer7(x) + x

        # 修改Conv2的连接方式
        output_layer5 = self.layer5(output_layer3)
        output_layer6 = self.layer6(output_layer5)
        output_layer7 = self.layer7(output_layer6 + output_layer5) + output_layer6 + output_layer5

        out = self.ca2(output_layer7) * output_layer7
        out = self.sa2(out) * out
        #output_layer7 = self.relu2(output_layer7)
        output_layer7 =out+output_layer7

        #Conv3
        # x = self.layer9(x)
        # x = self.layer10(x) + x
        # x = self.layer11(x) + x

        # 修改Conv3的连接方式
        output_layer9 = self.layer9(output_layer7)
        output_layer10 = self.layer10(output_layer9)
        output_layer11 = self.layer11(output_layer10 + output_layer9) + output_layer10 + output_layer9

        out = self.ca3(output_layer11) * output_layer11
        out = self.sa3(out) * out
        #output_layer11 = self.relu3(output_layer11)
        output_layer11 =out+output_layer11

        return output_layer11

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        #self.layer15 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(128, 128, kernel_size=3, padding=1)
        #    ) 
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #self.layer19 = nn.Sequential(
        #    nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #    )
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        #self.layer23 = nn.Sequential(
        #    nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #    nn.ReLU(),
        #    nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #    ) 
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
    def forward(self,x):        
        # #Deconv3
        # x = self.layer13(x) + x
        # x = self.layer14(x) + x
        # x = self.layer16(x)

        # 修改Deconv3的连接方式
        output_layer13 = self.layer13(x)
        output_layer13_add = output_layer13 + x
        output_layer14 = self.layer14(output_layer13_add)+output_layer13_add
        output_layer16 = self.layer16(output_layer14)

        # #Deconv2
        # x = self.layer17(x) + x
        # x = self.layer18(x) + x
        # x = self.layer20(x)

        # 修改Deconv2的连接方式
        output_layer17 = self.layer17(output_layer16)
        output_layer17_add = output_layer17 + output_layer16
        output_layer18 = self.layer18(output_layer17_add)+output_layer17_add
        output_layer20 = self.layer20(output_layer18)

        # #Deconv1
        # x = self.layer21(x) + x
        # x = self.layer22(x) + x
        # x = self.layer24(x)

        # 修改Conv1的连接方式
        output_layer21 = self.layer21(output_layer20)
        output_layer21_add = output_layer21 + output_layer20
        output_layer22 = self.layer22(output_layer21_add)+output_layer21_add
        output_layer24 = self.layer24(output_layer22)
        return output_layer24
