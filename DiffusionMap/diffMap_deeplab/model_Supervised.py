import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from diffMap_deeplab.diffMap_layers import *
import numpy as np
affine_par = True


#SpixelAggr_avr = SpixelAggr_avr.apply
#Feat2Dist = Feat2Dist.apply
#dist2SimiMatrix_scale = dist2SimiMatrix_scale.apply
#neighMasking = neighMasking.apply
#compNormSimiMatrix = compNormSimiMatrix.apply
#compEigDecomp = compEigDecomp.apply
#compDiffDist = compDiffDist.apply

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class PredBlock(nn.Module):
    def __init_(self):
        super(PredBlock, self).__init__()
        self.conf_diff = conf_diff()


    def forward(self,inputs):
        (x, W) = inputs
        out = self.conf_diff(x, W)

        return out

class DiffBlock(nn.Module):
    def __init__(self):
        super(DiffBlock, self).__init__()
        wei1 = 5e-3
        wei2 = 1e+3
        self.SpixelsAggr_avr = SpixelAggr_avr() # no learnable parameters
        self.Feat2Dist = Feat2Dist() # with learnable parameters
        #self.dist2SimiMatrix_scale = dist2SimiMatrix_scale() # with learnable weights
        self.dist2SimiMatrix1 = dist2SimiMatrix(wei1)  # with learnable weights
        self.neighMasking = neighMasking()  # no learnable parameters
        self.compNormSimiMatrix = compNormSimiMatrix()  # no learnable parameters
        self.compEigDecomp = compEigDecomp() # no learnable parameters
        self.compDiffDist = compDiffDist() # no learnable parameters
        self.dist2SimiMatrix2 = dist2SimiMatrix(wei2)  # with learnable weights


    def forward(self, inputs):
        (x, seg_labels, neig_mask, coor_idx) = inputs
        out = self.SpixelsAggr_avr(x, seg_labels, coor_idx)
        out = self.Feat2Dist(out)
        out = self.dist2SimiMatrix1(out)
        out = self.neighMasking(out, neig_mask)
        out = self.compNormSimiMatrix(out)
        out_G, out_V = self.compEigDecomp(out)
        out = self.compDiffDist(out_G, out_V)
        out = self.dist2SimiMatrix2(out)

        return out

class BasicBlock_noBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes1, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock_noBN, self).__init__()
        self.conv1 = conv3x3(inplanes1, planes[0], stride)
        #self.bn1 = nn.BatchNorm2d(planes[0], affine = affine_par)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes[0], planes[1], stride)
        #self.bn2 = nn.BatchNorm2d(planes[1], affine = affine_par)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(planes[1], planes[2], stride)
        #self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        # out = self.bn1(out)
        out = self.relu2(out)


        out = self.conv3(out)
        #out = self.bn2(out)

        #if self.downsample is not None:
        #    residual = self.downsample(x)

        #out += residual
        #out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class Residual_Covolution(nn.Module):
    def __init__(self, icol, ocol, num_classes):
        super(Residual_Covolution, self).__init__()
        self.conv1 = nn.Conv2d(icol, ocol, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv2 = nn.Conv2d(ocol, num_classes, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv3 = nn.Conv2d(num_classes, ocol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(ocol, icol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg

class Residual_Refinement_Module(nn.Module):

    def __init__(self, num_classes):
        super(Residual_Refinement_Module, self).__init__()
        self.RC1 = Residual_Covolution(2048, 512, num_classes)
        self.RC2 = Residual_Covolution(2048, 512, num_classes)

    def forward(self, x):
        x, seg1 = self.RC1(x)
        _, seg2 = self.RC2(x)
        return [seg1, seg1+seg2]

class ResNet_Refine(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet_Refine, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = Residual_Refinement_Module(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

class segNet_diff(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(segNet_diff, self).__init__()

        # subnet 1: diff
        self.conv1_net1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_net1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1_net1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1_net1 = self._make_layer(block, 64, layers[0])
        self.layer2_net1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_net1 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4_net1 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5_net1 = self._make_basic_layer(BasicBlock_noBN, 2048, [1024, 1024, 512], stride=1)
        self.layer6_net1 = self._make_diff_layer(DiffBlock)

        # subnet2: seg
        self.conv1_net2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)
        self.bn1_net2 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1_net2.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1_net2 = self._make_layer(block, 64, layers[0])
        self.layer2_net2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_net2 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4_net2 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5_net2 = self._make_basic_layer(BasicBlock_noBN, 2048, [1024, 1024, 512], stride=1)

        # subnet: conf_diff
        self.layer_conf_diff = self._make_diff_layer(PredBlock)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_diff_layer(self, block):
        layers = []
        layers.append(block())

        return nn.Sequential(*layers)

    def _make_basic_layer(self, block, planes, blocks, stride = 1):
        layers = []
        layers.append(block(self.inplanes, blocks, stride))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        if (downsample != None):
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        (x, neig_mask, seg_labels, coor_idx) = inputs
        y = x

        # subnet for diffusion distance map
        x = self.conv1_net1(x)
        x = self.bn1_net1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1_net1(x)
        x = self.layer2_net1(x)
        x = self.layer3_net1(x)
        x = self.layer4_net1(x)
        x = self.layer5_net1(x)
        x = self.layer6_net1((x, neig_mask, seg_labels, coor_idx))

        # subnet for deeplab based sementic segmentation.
        y = self.conv1_net2(y)
        y = self.bn1_net2(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.layer1_net2(y)
        y = self.layer2_net2(y)
        y = self.layer3_net2(y)
        y = self.layer4_net2(y)
        y = self.layer5_net2(y)

        # subnet for predicting final labels
        y = self.layer_conf_diff(x,y)

        return x, y




class DiffDistance_net(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(DiffDistance_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_basic_layer(BasicBlock_noBN, 2048, [1024, 1024, 512], stride=1)
        self.layer6 = self._make_diff_layer(DiffBlock)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_diff_layer(self, block):
        layers = []
        layers.append(block())

        return nn.Sequential(*layers)

    def _make_basic_layer(self, block, planes, blocks, stride = 1):
        layers = []
        layers.append(block(self.inplanes, blocks, stride))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        if (downsample != None):
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        (x, neig_mask, seg_labels, coor_idx) = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6((x, neig_mask, seg_labels, coor_idx))

        return x






class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

class MS_Deeplab(nn.Module):
    def __init__(self,block,num_classes):
        super(MS_Deeplab,self).__init__()
        self.Scale = ResNet(block,[3, 4, 23, 3],num_classes)   #changed to fix #4 

    def forward(self,x):
        output = self.Scale(x) # for original scale
        output_size = output.size()[2]
        input_size = x.size()[2]

        self.interp1 = nn.Upsample(size=(int(input_size*0.75)+1, int(input_size*0.75)+1), mode='bilinear')
        self.interp2 = nn.Upsample(size=(int(input_size*0.5)+1, int(input_size*0.5)+1), mode='bilinear')
        self.interp3 = nn.Upsample(size=(output_size, output_size), mode='bilinear')

        x75 = self.interp1(x)
        output75 = self.interp3(self.Scale(x75)) # for 0.75x scale

        x5 = self.interp2(x)
        output5 = self.interp3(self.Scale(x5))	# for 0.5x scale

        out_max = torch.max(torch.max(output, output75), output5)
        return [output, output75, output5, out_max]


def DiffNet():
    model = DiffDistance_net(Bottleneck, [3, 4, 23, 3])
    return model

def Res_Ms_Deeplab(num_classes=21):
    model = MS_Deeplab(Bottleneck, num_classes)
    return model

def Res_Deeplab(num_classes=21, is_refine=False):
    if is_refine:
        model = ResNet_Refine(Bottleneck,[3, 4, 23, 3], num_classes)
    else:
        model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model
