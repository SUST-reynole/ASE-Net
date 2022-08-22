import torch.nn as nn
import torch
from torch.nn import init
from torch.nn.functional import interpolate
from thop import *

import torch.nn.functional as F
class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        #assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_planes)
        
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
       
        x1 = self.avgpool(x) # b c 1 1 
        
        x2 = self.fc1(x1) #b k 1 1

        x3 = F.relu(x2)
        x4 = self.fc2(x3).view(x3.size(0), -1) # b k
        return F.softmax(x4/self.temperature, 1),torch.sigmoid(x1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=False, K=4,temperature=30, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.groups = min(in_planes,out_planes)
        self.groups = in_planes
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)
        # self.finall = nn.BatchNorm2d(out_planes)
        # self.rl = nn.ReLU(inplace=True)
        self.weight = nn.Parameter(torch.Tensor(K,in_planes, in_planes//self.groups, kernel_size, kernel_size), requires_grad=True)
        self.conv_1 = nn.Conv2d(in_planes,out_planes,1)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()
       
        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的#2 3 512 512
        softmax_attention,sigmiod_atten = self.attention(x)
        x = sigmiod_atten*x
        batch_size, _, height, width = x.size()
        # 
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        
        weight = self.weight.view(self.K, -1) #4 432   # K output input 3 3 

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同)
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size) #分组卷积将权重和特征图都进行分组

        output = output.view(batch_size, self.in_planes, output.size(-2), output.size(-1))
        output_f = self.conv_1(output)
        # output = self.finall(output)
        # output = self.rl(output)
        return output_f



class sSE_Module(nn.Module):
    def __init__(self, channel):
        super(sSE_Module, self).__init__()
        self.spatial_excitation = nn.Sequential(
                nn.Conv2d(in_channels=channel, out_channels=channel//16, kernel_size=1,stride=1,padding=0),
                nn.Conv2d(in_channels=channel//16, out_channels=1, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
    def forward(self, x):
        z = self.spatial_excitation(x)
        return x * z.expand_as(x)
class DYBAC(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,stride, padding):
        super(DYBAC, self).__init__()
        self.dybac = nn.Sequential(
           sSE_Module(in_planes),
           Dynamic_conv2d(in_planes=in_planes, out_planes=out_planes,kernel_size=kernel_size,stride=stride,padding=padding), 
        #    sSE_Module(out_planes),
        )
    def forward(self, x):
        z = self.dybac(x)
        return  z
class DoubleConv1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
        )
        

    def forward(self, input):
        return self.conv(input)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
           
            DYBAC(in_ch, out_ch, 3, padding=1,stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
            DYBAC(out_ch, out_ch, 3, padding=1,stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
           
         
        )
        
    def forward(self, input):
        return self.conv(input)



class Unet_dy_v2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_dy_v2, self).__init__()

        self.conv1 = DoubleConv1(in_ch, 64)#1
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.1)
        self.conv2 = DoubleConv(64, 128)#2
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.1)
        self.conv3 = DoubleConv(128, 256)#3
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(0.2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(0.2)
        self.conv5 = DoubleConv(512, 1024)
        self.drop5 = nn.Dropout2d(0.3)
        
        self.conv6 = DoubleConv(1536, 512)
        self.drop6 = nn.Dropout2d(0.2)
        
        self.conv7 = DoubleConv(768, 256)
        self.drop7 = nn.Dropout2d(0.2)
     
        self.conv8 = DoubleConv(384, 128)
        self.drop8 = nn.Dropout2d(0.1)
     
        self.conv9 = DoubleConv(192, 64)
        self.drop9 = nn.Dropout2d(0.1)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        d1 = self.drop1(p1)
        c2 = self.conv2(d1)
        p2 = self.pool2(c2)
        d2 = self.drop2(p2)
        c3 = self.conv3(d2)
        p3 = self.pool3(c3)
        d3 = self.drop3(p3)
        c4 = self.conv4(d3)
        p4 = self.pool4(c4)
        d4 = self.drop4(p4)
        c5 = self.conv5(d4)
        d5 = self.drop5(c5)
        up_6 = nn.Upsample(scale_factor=2)(d5)
        # up_6 = interpolate(d5,scale_factor=2,mode="bilinear")
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        d6 = self.drop6(c6)
        up_7 = nn.Upsample(scale_factor=2)(d6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        d7 = self.drop7(c7)
        up_8 = nn.Upsample(scale_factor=2)(d7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        d8 = self.drop8(c8)
        up_9 = nn.Upsample(scale_factor=2)(d8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        d9 = self.drop9(c9)
        c10 = self.conv10(d9)
        # out = nn.Sigmoid()(c10)
        return c10
if __name__ == "__main__":

        
    input = torch.rand(1, 1, 256, 256).cuda()
    model = Unet_dy_v2(1, 1).cuda()
   


    
   
    flops1, params1 = profile(model, (input,))
    flops1, params1 = clever_format([flops1, params1], "%.3f")#将数据换算为G以及MB的函数
    print(params1,flops1)

