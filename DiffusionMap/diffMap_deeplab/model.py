import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from diffMap_deeplab.diffMap_layers_f import *
import numpy as np
affine_par = True
from deeplab.model import SelfAttention_Module_MS, diffAttention_MS



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


class DiffBlock(nn.Module):
    def __init__(self):
        super(DiffBlock, self).__init__()
        wei1 = 3.3546e-04 #5e-3
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

class DiffBlock_grid_rs(nn.Module):
    def __init__(self):
        super(DiffBlock_grid_rs, self).__init__()
        wei1 = 1e-3
        wei2 = 1e+3 #4e+3
        self.Feat2Dist = Feat2Dist_batch() # with learnable parameters
        #self.dist2SimiMatrix_scale = dist2SimiMatrix_scale() # with learnable weights
        self.dist2SimiMatrix1 = dist2SimiMatrix_batch(wei1)  # with learnable weights
        self.neighMasking = neighMasking_batch()  # no learnable parameters
        self.compNormSimiMatrix = compNormSimiMatrix_batch()  # no learnable parameters
        self.compEigDecomp = compEigDecomp_batch() # no learnable parameters
        self.compDiffDist = compDiffDist_batch() # no learnable parameters
        self.dist2SimiMatrix2 = dist2SimiMatrix_batch(wei2)  # with learnable weights

        self.featUpsample = torch.nn.Upsample(size = (41,41), mode = 'bilinear')


    def forward(self, inputs):
        (x, neig_mask) = inputs
        #out = self.SpixelsAggr_avr(x, seg_labels, coor_idx)
#       a = time.time()
        x = self.featUpsample(x) #resize the input feature map. #torch.nn.Upsample(size=(50, 50), mode='bilinear')(torch.rand(1, 3, 64, 64))
        out = self.Feat2Dist(x) # 0.01s

#        b1 = time.time() # 0.32s
        out = self.dist2SimiMatrix1(out) #0.008s
        out = self.neighMasking(out, neig_mask) #0.001s

        out = self.compNormSimiMatrix(out) #0.14s

#        b2 = time.time()
        out_G, out_V = self.compEigDecomp(out) # 0.41s

#        b3 = time.time()
        out = self.compDiffDist(out_G, out_V) # 0.05s

#        b4 = time.time()
        out = self.dist2SimiMatrix2(out) #0.015s

#        print('stamp2s:', b1 - a, b2 - b1, b3 - b2, b4 - b3)

        return out

class MLBlock_grid_rs(nn.Module):
    def __init__(self):
        super(MLBlock_grid_rs, self).__init__()
        wei1 = 1e-3
        wei2 = 1e+3 #4e+3
        self.Feat2Dist = Feat2Dist_batch() # with learnable parameters
        #self.dist2SimiMatrix_scale = dist2SimiMatrix_scale() # with learnable weights
        self.dist2SimiMatrix1 = dist2SimiMatrix_batch(wei1)  # with learnable weights
        self.neighMasking = neighMasking_batch()  # no learnable parameters
        self.compNormSimiMatrix = compNormSimiMatrix_batch()  # no learnable parameters
        self.compEigDecomp = compEigDecomp_batch() # no learnable parameters
        self.compDiffDist = compDiffDist_batch() # no learnable parameters
        self.dist2SimiMatrix2 = dist2SimiMatrix_batch(wei2)  # with learnable weights

        self.featUpsample = torch.nn.Upsample(size = (41,41), mode = 'bilinear')


    def forward(self, inputs):
        (x, neig_mask) = inputs
        #out = self.SpixelsAggr_avr(x, seg_labels, coor_idx)
#       a = time.time()
        x = self.featUpsample(x) #resize the input feature map. #torch.nn.Upsample(size=(50, 50), mode='bilinear')(torch.rand(1, 3, 64, 64))
        out = self.Feat2Dist(x) # 0.01s

#        b1 = time.time() # 0.32s
        out = self.dist2SimiMatrix1(out) #0.008s

        #out = self.neighMasking(out, neig_mask) #0.001s

        #out = self.compNormSimiMatrix(out) #0.14s

#        b2 = time.time()
        #out_G, out_V = self.compEigDecomp(out) # 0.41s

#        b3 = time.time()
        #out = self.compDiffDist(out_G, out_V) # 0.05s

#        b4 = time.time()
        #out = self.dist2SimiMatrix2(out) #0.015s

#        print('stamp2s:', b1 - a, b2 - b1, b3 - b2, b4 - b3)

        return out

class Upsample_fullScale(nn.Module):
    def __init__(self):
        super(Upsample_fullScale, self).__init__()
        self.weights = nn.Parameter(torch.Tensor([8, 4, 1]))
        #self.inteFeat = inter_feat_linear()


    def forward(self, FeatMaps, lab):
        ## inputs:
        # label_map: N * D * W * H, D is the number of clusters / categories
        # FeatMaps_8X: N * C1 * W/8 * H/8, C1 is the number of feature maps
        # FeatMaps_4X: N * C2 * W/4 * H/4, C2 is the number of feature maps
        # FeatMaps_2X: N * C3 * W/2 * H/2, C3 is the number of feature maps
        # FeatMaps_1X: N * C4 * W   * H,   C4 is the number of feature maps

        # labMap: input label maps with size of 41 * 41, with size of N * D * 41 * 41
        ## output: the upsampled labMap, with size of N * D, W * H
        nlv = FeatMaps.__len__()
        inteFeat = inter_feat_linear()
        # upsample at multipe upsampling scale
        nCls = lab.size(1)
        lab_curr = lab
        for sc in range(nlv): #nlv
            FeatMap = FeatMaps[sc]

            with torch.no_grad():
                r = FeatMap.size(2)
                c = FeatMap.size(3)
                r_lb = lab_curr.size(2)
                c_lb = lab_curr.size(3)

                inteX = (r - 1)/(r_lb - 1)
                inteY = (c - 1)/(c_lb - 1)

                grid = torch.zeros(1, r,c, 2).cuda()
                grid_lb = torch.zeros(1, r_lb, c_lb, 2).cuda()

                x = torch.range(0.0, r - 1, 1)
                y = torch.range(0.0, c - 1, 1)
                grid[0,:,:,0] = x.reshape(-1, 1).repeat(1,y.size(0))
                grid[0,:,:,1] = y.reshape(1,-1).repeat(x.size(0), 1)

                x_lb = torch.range(0.0, r - 1 + inteX / 2, inteX)
                y_lb = torch.range(0.0, c - 1 + inteY / 2, inteY)

                try:
                    grid_lb[0,:, :, 0] = x_lb.reshape(-1, 1).repeat(1, y_lb.size(0))
                    grid_lb[0,:, :, 1] = y_lb.reshape(1, -1).repeat(x_lb.size(0), 1)
                except:
                    print(r, c, r_lb, c_lb)

            lab_curr = inteFeat(grid, grid_lb, FeatMap, lab_curr, self.weights[sc])
            #Upsample = torch.nn.Upsample(size=(r, c), mode='bilinear')
            #lab_curr = Upsample(lab_curr)

        return lab_curr

class Upsample_fullScale_cpu(nn.Module):
    def __init__(self):
        super(Upsample_fullScale_cpu, self).__init__()
        self.weights = nn.Parameter(torch.Tensor([8, 4, 1]))
        #self.inteFeat = inter_feat_linear()


    def forward(self, FeatMaps, lab):
        ## inputs:
        # label_map: N * D * W * H, D is the number of clusters / categories
        # FeatMaps_8X: N * C1 * W/8 * H/8, C1 is the number of feature maps
        # FeatMaps_4X: N * C2 * W/4 * H/4, C2 is the number of feature maps
        # FeatMaps_2X: N * C3 * W/2 * H/2, C3 is the number of feature maps
        # FeatMaps_1X: N * C4 * W   * H,   C4 is the number of feature maps

        # labMap: input label maps with size of 41 * 41, with size of N * D * 41 * 41
        ## output: the upsampled labMap, with size of N * D, W * H
        nlv = FeatMaps.__len__()
        inteFeat = inter_feat_linear_cpu()
        # upsample at multipe upsampling scale
        nCls = lab.size(1)
        lab_curr = lab
        for sc in range(nlv): #nlv
            FeatMap = FeatMaps[sc].cpu()

            with torch.no_grad():
                r = FeatMap.size(2)
                c = FeatMap.size(3)
                r_lb = lab_curr.size(2)
                c_lb = lab_curr.size(3)

                inteX = r/r_lb
                inteY = c/c_lb

                grid = torch.zeros(1, r,c, 2)
                grid_lb = torch.zeros(1, r_lb, c_lb, 2)

                x = torch.range(0.0, r - 1, 1)
                y = torch.range(0.0, c - 1, 1)
                grid[0,:,:,0] = x.reshape(-1, 1).repeat(1,y.size(0))
                grid[0,:,:,1] = y.reshape(1,-1).repeat(x.size(0), 1)

                x_lb = torch.range(0.0, r - 1, inteX)
                y_lb = torch.range(0.0, c - 1, inteY)
                grid_lb[0,:, :, 0] = x_lb.reshape(-1, 1).repeat(1, y_lb.size(0))
                grid_lb[0,:, :, 1] = y_lb.reshape(1, -1).repeat(x_lb.size(0), 1)

            lab_curr = inteFeat(grid, grid_lb, FeatMap, lab_curr, self.weights[sc])
            #Upsample = torch.nn.Upsample(size=(r, c), mode='bilinear')
            #lab_curr = Upsample(lab_curr)

        return lab_curr


class DiffBlock_grid(nn.Module):
    def __init__(self):
        super(DiffBlock_grid, self).__init__()
        wei1 = 1e-3 #1e-3
        wei2 = 1e+3 #1e+3
        self.Feat2Dist = Feat2Dist_batch() # with learnable parameters
        #self.dist2SimiMatrix_scale = dist2SimiMatrix_scale() # with learnable weights
        self.dist2SimiMatrix1 = dist2SimiMatrix_batch(wei1)  # with learnable weights
        self.neighMasking = neighMasking_batch()  # no learnable parameters
        self.compNormSimiMatrix = compNormSimiMatrix_batch()  # no learnable parameters
        self.compEigDecomp = compEigDecomp_batch() # no learnable parameters
        self.compDiffDist = compDiffDist_batch() # no learnable parameters
        self.dist2SimiMatrix2 = dist2SimiMatrix_batch(wei2)  # with learnable weights

        self.featUpsample = torch.nn.Upsample(size = (41,41), mode = 'bilinear')


    def forward(self, inputs):
        (x, neig_mask) = inputs # si indicates the feature map size
        #out = self.SpixelsAggr_avr(x, seg_labels, coor_idx)
#       a = time.time()
        x = self.featUpsample(x) #resize the input feature map. #torch.nn.Upsample(size=(50, 50), mode='bilinear')(torch.rand(1, 3, 64, 64))

        out = self.Feat2Dist(x) # 0.01s
        si = [41,41]#[x.size(2), x.size(3)]

#        b1 = time.time() # 0.32s
        out = self.dist2SimiMatrix1(out) #0.008s
        out_simi = out

        out = self.neighMasking(out, neig_mask) #0.001s

        out = self.compNormSimiMatrix(out) #0.14s

#        b2 = time.time()
        out_G, out_V = self.compEigDecomp(out, si) # 0.41s

#        b3 = time.time()
        out = self.compDiffDist(out_G, out_V) # 0.05s

#        b4 = time.time()
        out = self.dist2SimiMatrix2(out) #0.015s

#        print('stamp2s:', b1 - a, b2 - b1, b3 - b2, b4 - b3)

        return out, out_simi

class MLBlock_grid(nn.Module):
    def __init__(self):
        super(MLBlock_grid, self).__init__()
        wei1 = 1e-3 #1e-3
        wei2 = 1e+3 #1e+3
        self.Feat2Dist = Feat2Dist_batch() # with learnable parameters
        #self.dist2SimiMatrix_scale = dist2SimiMatrix_scale() # with learnable weights
        self.dist2SimiMatrix1 = dist2SimiMatrix_batch(wei1)  # with learnable weights
        self.neighMasking = neighMasking_batch()  # no learnable parameters
        self.compNormSimiMatrix = compNormSimiMatrix_batch()  # no learnable parameters
        self.compEigDecomp = compEigDecomp_batch() # no learnable parameters
        self.compDiffDist = compDiffDist_batch() # no learnable parameters
        self.dist2SimiMatrix2 = dist2SimiMatrix_batch(wei2)  # with learnable weights

        self.featUpsample = torch.nn.Upsample(size = (41,41), mode = 'bilinear')


    def forward(self, inputs):
        (x, neig_mask) = inputs # si indicates the feature map size
        #out = self.SpixelsAggr_avr(x, seg_labels, coor_idx)
#       a = time.time()
        x = self.featUpsample(x) #resize the input feature map. #torch.nn.Upsample(size=(50, 50), mode='bilinear')(torch.rand(1, 3, 64, 64))

        out = self.Feat2Dist(x) # 0.01s
        si = [41,41]#[x.size(2), x.size(3)]

#        b1 = time.time() # 0.32s
        out = self.dist2SimiMatrix1(out) #0.008s
        out_simi = out

        #out = self.neighMasking(out, neig_mask) #0.001s

        #out = self.compNormSimiMatrix(out) #0.14s

#        b2 = time.time()
        #out_G, out_V = self.compEigDecomp(out, si) # 0.41s

#        b3 = time.time()
        #out = self.compDiffDist(out_G, out_V) # 0.05s

#        b4 = time.time()
        #out = self.dist2SimiMatrix2(out) #0.015s

#        print('stamp2s:', b1 - a, b2 - b1, b3 - b2, b4 - b3)

        return out, out_simi

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

    def __init__(self, inplanes, planes, stride=1, dilation=1, moment=0, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par, momentum=moment)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par, momentum=moment)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par, momentum=moment)
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

'''
# kernel kmeans algorithm at training time
class Kernel_KMeans_Train(nn.Module):   # at training time, g.t. clusters are given
    def __init__(self):
        super(Kernel_KMeans_Train, self).__init__()

    def forward(self, feat, feat_orig, gt_cluster):
        # extract centers from gt_clusters
        

        # perform kernel kmeans by running multiple times


        # cluster reassignment for


        return Prob_cls
'''

'''
# kernel kmeans algorithm at testing time
class Kernel_KMeans_Test(nn.Module):   # at test time, g.t. clusters are not given, we should initialize clusters centers randomnly
'''



class DiffDistance_atgrid_net(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(DiffDistance_grid_net, self).__init__()
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
        self.layer5 = self._make_basic_layer(BasicBlock_noBN, 2368, [1024, 1024, 1024], stride=1) #2048
        self.layer6 = self._make_diff_layer(DiffBlock_grid)  #_rs
        self.avrpool1 = nn.AvgPool2d(kernel_size=17, stride=8, padding=8)
        self.avrpool2 = nn.AvgPool2d(kernel_size=9, stride=4, padding=4)
        self.avrpool3 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

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
        self.inplanes = planes
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
        #a = time.time()

        (im, neig_mask) = inputs
        x = self.conv1(im)
        x = self.bn1(x)
        x = self.relu(x)
        x_con1 = x
        x = self.maxpool(x)
        x = self.layer1(x)
        x_con2 = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(torch.cat((x, self.avrpool2(x_con1), self.avrpool3(x_con2)), 1))
        #x = self.layer5(x)

        #x = self.layer6((torch.cat((x, self.avrpool2(x_con1), self.avrpool3(x_con2)), 1), neig_mask))
        x, x_simi = self.layer6((x, neig_mask))


        #b2 = time.time()
        #print('stamp2:', b2 - a)

        return x, x_simi

class DiffDistance_grid_net(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(DiffDistance_grid_net, self).__init__()
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
        self.layer5 = self._make_basic_layer(BasicBlock_noBN, 2368, [1024, 1024, 1024], stride=1) #2048
        self.layer6 = self._make_diff_layer(DiffBlock_grid)  #
        self.avrpool1 = nn.AvgPool2d(kernel_size=17, stride=8, padding=8)
        self.avrpool2 = nn.AvgPool2d(kernel_size=9, stride=4, padding=4)
        self.avrpool3 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

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
        self.inplanes = planes
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
        #a = time.time()

        (im, neig_mask) = inputs
        x = self.conv1(im)
        x = self.bn1(x)
        x = self.relu(x)

        x_con1 = x

        x = self.maxpool(x)
        x = self.layer1(x)

        x_con2 = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_con3 = x


        ## feature concatenation
        con1_pool = self.avrpool2(x_con1)
        con2_pool = self.avrpool3(x_con2)

        if 1:
            with torch.no_grad():
                si0 = x_con3.shape[2:4]
                si1 = con1_pool.shape[2:4]
                si2 = con2_pool.shape[2:4]

            if not (si0 == si1 == si2):
                # make the feature map in same size
                featUpsample = torch.nn.Upsample(size=si0, mode='bilinear')
                if not(si1 == si0):
                    con1_pool = featUpsample(con1_pool)
                if not(si2 == si0):
                    con2_pool = featUpsample(con2_pool)

        x = self.layer5(torch.cat((x_con3, con1_pool, con2_pool), 1))

        #x = self.layer5(x)
        #if x_con1.size(2) < x_con1.size(1)
        #x.size(3)

        #x_con1.size(2)
        #x_con1.size(3)

        #x = self.layer6((torch.cat((x, self.avrpool2(x_con1), self.avrpool3(x_con2)), 1), neig_mask))
        x, x_simi = self.layer6((x, neig_mask))

        #b2 = time.time()
        #print('stamp2:', b2 - a)

        return x, x_simi, (x_con3, x_con2, x_con1)

class DFeat_grid_net(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(DFeat_grid_net, self).__init__()
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
        self.layer5 = self._make_basic_layer(BasicBlock_noBN, 2368, [1024, 1024, 1024], stride=1) #2048
        self.layer6 = self._make_diff_layer(DiffBlock_grid)  #
        self.avrpool1 = nn.AvgPool2d(kernel_size=17, stride=8, padding=8)
        self.avrpool2 = nn.AvgPool2d(kernel_size=9, stride=4, padding=4)
        self.avrpool3 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

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
        self.inplanes = planes
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
        #a = time.time()

        (im, neig_mask) = inputs
        x = self.conv1(im)
        x = self.bn1(x)
        x = self.relu(x)

        x_con1 = x

        x = self.maxpool(x)
        x = self.layer1(x)

        x_con2 = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_con3 = x


        ## feature concatenation
        con1_pool = self.avrpool2(x_con1)
        con2_pool = self.avrpool3(x_con2)

        if 1:
            with torch.no_grad():
                si0 = x_con3.shape[2:4]
                si1 = con1_pool.shape[2:4]
                si2 = con2_pool.shape[2:4]

            if not (si0 == si1 == si2):
                # make the feature map in same size
                featUpsample = torch.nn.Upsample(size=si0, mode='bilinear')
                if not(si1 == si0):
                    con1_pool = featUpsample(con1_pool)
                if not(si2 == si0):
                    con2_pool = featUpsample(con2_pool)

        x = self.layer5(torch.cat((x_con3, con1_pool, con2_pool), 1))

        #x = self.layer5(x)
        #if x_con1.size(2) < x_con1.size(1)
        #x.size(3)

        #x_con1.size(2)
        #x_con1.size(3)

        #x = self.layer6((torch.cat((x, self.avrpool2(x_con1), self.avrpool3(x_con2)), 1), neig_mask))
        #x, x_simi = self.layer6((x, neig_mask))

        #b2 = time.time()
        #print('stamp2:', b2 - a)

        return x #, x_simi, (x_con3, x_con2, x_con1)


class MetricLearning_grid_net(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(MetricLearning_grid_net, self).__init__()
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
        self.layer5 = self._make_basic_layer(BasicBlock_noBN, 2368, [1024, 1024, 1024], stride=1) #2048
        self.layer6 = self._make_diff_layer(MLBlock_grid)  #DiffBlock_grid
        self.avrpool1 = nn.AvgPool2d(kernel_size=17, stride=8, padding=8)
        self.avrpool2 = nn.AvgPool2d(kernel_size=9, stride=4, padding=4)
        self.avrpool3 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

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
        self.inplanes = planes
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
        #a = time.time()

        (im, neig_mask) = inputs
        x = self.conv1(im)
        x = self.bn1(x)
        x = self.relu(x)

        x_con1 = x

        x = self.maxpool(x)
        x = self.layer1(x)

        x_con2 = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_con3 = x


        ## feature concatenation
        con1_pool = self.avrpool2(x_con1)
        con2_pool = self.avrpool3(x_con2)

        if 1:
            with torch.no_grad():
                si0 = x_con3.shape[2:4]
                si1 = con1_pool.shape[2:4]
                si2 = con2_pool.shape[2:4]

            if not (si0 == si1 == si2):
                # make the feature map in same size
                featUpsample = torch.nn.Upsample(size=si0, mode='bilinear')
                if not(si1 == si0):
                    con1_pool = featUpsample(con1_pool)
                if not(si2 == si0):
                    con2_pool = featUpsample(con2_pool)

        x = self.layer5(torch.cat((x_con3, con1_pool, con2_pool), 1))

        #x = self.layer5(x)
        #if x_con1.size(2) < x_con1.size(1)
        #x.size(3)

        #x_con1.size(2)
        #x_con1.size(3)

        #x = self.layer6((torch.cat((x, self.avrpool2(x_con1), self.avrpool3(x_con2)), 1), neig_mask))
        x, x_simi = self.layer6((x, neig_mask))

        #b2 = time.time()
        #print('stamp2:', b2 - a)

        return x, x_simi, (x_con3, x_con2, x_con1)


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
        self.layer6 = self._make_diff_layer(DiffBlock)  #


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


class ResNet_diff_seg_semisupervised(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(DiffDistance_grid_net, self).__init__()
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
        self.layer5_diff = self._make_basic_layer(BasicBlock_noBN, 2368, [1024, 1024, 1024], stride=1) #2048
        self.layer6_diff = self._make_diff_layer(DiffBlock_grid)  #
        self.layer5_seg = self._make_atten_layer(SelfAttention_Module_MS, 1, num_classes, [1], [1])

        self.avrpool1 = nn.AvgPool2d(kernel_size=17, stride=8, padding=8)
        self.avrpool2 = nn.AvgPool2d(kernel_size=9, stride=4, padding=4)
        self.avrpool3 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

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

    def _make_atten_layer(self,block, stride, num_classes, dilation_series, padding_series):
        return block(stride,num_classes,dilation_series, padding_series)

    def _make_basic_layer(self, block, planes, blocks, stride = 1):
        layers = []
        self.inplanes = planes
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
        #a = time.time()

        (im, neig_mask) = inputs
        x = self.conv1(im)
        x = self.bn1(x)
        x = self.relu(x)

        x_con1 = x

        x = self.maxpool(x)
        x = self.layer1(x)

        x_con2 = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_con3 = x

        ###### SUB-Block for computing diffusion net
        x_diff = self.layer5_diff(torch.cat((x_con3, self.avrpool2(x_con1), self.avrpool3(x_con2)), 1))
        x_diff, x_diff_simi = self.layer6_diff((x_diff, neig_mask))

        ####### SUB-Block for computing segmentation label
        x, wei_cum, wei = self.layer5_seg(x, x_diff)

        return x, (x_con3, x_con2, x_con1)


class ResNet_diff_seg_supervised(nn.Module):
    def __init__(self, block, layers, num_classes, moment=0):
        self.inplanes = 64
        super(ResNet_diff_seg_supervised, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par, momentum=moment)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5_diff = self._make_basic_layer(BasicBlock_noBN, 2368, [1024, 1024, 1024], stride=1) #2048
        self.layer6_diff = self._make_diff_layer(DiffBlock_grid)  #
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        #self.layer5_seg = self._make_atten_layer(diffAttention_MS, 1, num_classes, [1], [1])

        self.avrpool1 = nn.AvgPool2d(kernel_size=17, stride=8, padding=8)
        self.avrpool2 = nn.AvgPool2d(kernel_size=9, stride=4, padding=4)
        self.avrpool3 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

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

    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def _make_basic_layer(self, block, planes, blocks, stride = 1):
        layers = []
        self.inplanes = planes
        layers.append(block(self.inplanes, blocks, stride))

        return nn.Sequential(*layers)

    def _make_atten_layer(self, block, stride, num_classes, dilation_series, padding_series):
        return block(stride, num_classes, dilation_series, padding_series)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, moment=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par, momentum=moment))
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
        #a = time.time()

        (im, neig_mask) = inputs
        x = self.conv1(im)
        x = self.bn1(x)
        x = self.relu(x)

        x_con1 = x

        x = self.maxpool(x)
        x = self.layer1(x)

        x_con2 = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_con3 = x

        ###### SUB-Block for computing diffusion net
        x_diff = self.layer5_diff(torch.cat((x_con3, self.avrpool2(x_con1), self.avrpool3(x_con2)), 1))
        x_diff, x_diff_simi = self.layer6_diff((x_diff, neig_mask))

        ####### SUB-Block for computing segmentation label
        x = self.layer5(x)

        #return x, x_simi, (x_con3, x_con2, x_con1)
        return x , x_diff, (x_con3, x_con2, x_con1)

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

def DiffNet_grid():
    model = DiffDistance_grid_net(Bottleneck, [3, 4, 23, 3])
    return model

def DFeat_grid():
    model = DFeat_grid_net(Bottleneck, [3, 4, 23, 3])  #DiffDistance_grid_net(Bottleneck, [3, 4, 23, 3])
    return model

def MetricLearning_grid():
    model = MetricLearning_grid_net(Bottleneck, [3, 4, 23, 3])
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

# implement supervised semantic segmentation net with two subnets sharing same feature extractor, two subnets learn diffusion map and per-pixel feature / label respectively
def Res_Deeplab_diffmap_supervised(num_classes=21, isatten=False):
    #if isatten:
    model = ResNet_diff_seg_supervised(Bottleneck,[3, 4, 23, 3], num_classes)
    #else:
    #    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model

# implement semi-supervised semantic segmentation net with two subnets sharing same feature extractor, two subnets learn diffusion map and per-pixel feature / label respectively
def Res_Deeplab_diffmap_semisupervised(num_classes=21, isatten=False):
    #if isatten:
    model = ResNet_diff_seg_semisupervised(Bottleneck,[3, 4, 23, 3], num_classes)
    #else:
    #    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model


