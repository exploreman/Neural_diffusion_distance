import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
affine_par = True


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par, momentum=0.8)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par, momentum=0.8)
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
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par, momentum=0.8)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par, momentum=0.8)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par, momentum=0.8)
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


class SelfAttention_Module_v2(nn.Module):

    def __init__(self, n_stride, num_classes):

        '''
        def __init__(self, dilation_series, padding_series, num_classes):
            super(Classifier_Module, self).__init__()
            self.conv2d_list = nn.ModuleList()
            for dilation, padding in zip(dilation_series, padding_series):
                self.conv2d_list.append(
                    nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                              bias=True))

            for m in self.conv2d_list:
                m.weight.data.normal_(0, 0.01)

        def forward(self, x):
            out = self.conv2d_list[0](x)
            for i in range(len(self.conv2d_list) - 1):
                out += self.conv2d_list[i + 1](x)
            return out

        '''


        super(SelfAttention_Module_v2, self).__init__()
        self.conv1 = nn.Conv2d(2048, num_classes, kernel_size=1, stride=n_stride, bias=True)  # change
        self.sig1 = nn.Sigmoid()

        #self.conv2 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=n_stride, bias=False)
        #self.softmax1 = nn.Softmax(dim=2)
        #self.conv3 = nn.Conv2d(2048, 1, kernel_size=1, stride=n_stride, bias=False)

        self.sig2 = nn.Sigmoid()
        self.weiPooling = weiPooling(num_classes)

        self.pred = pred_sw()
        self.softmax2 = nn.Softmax(dim=1)

        self.conv1.weight.data.normal_(0, 0.01)
        #nn.init.constant_(self.conv2.weight.data, 0)

        #self.conv3.weight.data.normal_(0, 0.01)
        #self.conv2.weight.data.normal_(0, 0.01)

        self.weight = nn.Parameter(torch.tensor([5.0]))
        #self.weights2 = nn.Parameter(torch.ones(num_classes, 2048) * 0.01).cuda()

    def forward(self, x, diffW):
        bs, d, w, h = x.size()
        out_diff = x.reshape(bs,d,w*h).bmm(diffW).reshape(bs, d, w, h)


        # prediction
        out = self.conv1(out_diff)
        #out = out.div(wnorm.permute(1,0,2,3).expand_as(out))
        out = out.reshape(out.size(0), out.size(1), -1)                         # out: bs, w*h, ncls

        # scaling before softmax
        out = self.softmax2(self.weight * out)
        out, wei = self.weiPooling(out, out_diff)

        # prediction
        out = self.pred(out, self.conv1.weight, self.conv1.bias)
        #out = out.div(wnorm.squeeze().expand_as(out))

        # transform to prob.
        bs, ncls = out.size()
        out = self.sig2(self.weight * out).reshape(bs, -1, ncls)

        if (out != out).sum():
            print(out)

        return out, wei

class diffAttention_MS(nn.Module):

    def __init__(self, n_stride, num_classes, dilation_series, padding_series):
        super(diffAttention_MS, self).__init__()

        self.ncls = num_classes
        self.conv2d_list = nn.ModuleList()
        self.conv1_list = nn.ModuleList()
        #self.sig1_list = nn.ModuleList()
        #self.sig2_list = nn.ModuleList()
        #self.weiPooling_list = nn.ModuleList()
        #self.pred_list = nn.ModuleList()
        #self.sofMax_list = nn.ModuleList()

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))
            self.conv1_list.append(nn.Conv2d(1024, num_classes, kernel_size=1, stride=n_stride, bias=True))  # change
            #self.sig1_list.append(nn.Sigmoid())
            #self.sig2_list.append(nn.Sigmoid())
            #self.sofMax_list.append(nn.Softmax(dim=1))
            #self.weiPooling_list.append(weiPooling(num_classes))
            #self.pred_list.append(pred_sw())

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

        for m in self.conv1_list:
            m.weight.data.normal_(0, 0.01)

        #self.sig1 = nn.Sigmoid()
        #self.sig2 = nn.Sigmoid()
        #self.weiPooling = weiPooling(num_classes)
        #self.pred = pred_sw()
        #self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x, diffW):

        bs, d, w, h = x.size()
        dim = 1024
        nit = len(self.conv2d_list)
        #out_diff_all = [] #torch.zeros(nit, bs, dim, w, h).cuda()
        #prob = torch.zeros(bs, self.ncls).cuda() #1,
        pred = torch.zeros(bs, self.ncls, w * h).cuda()

        for it in range(len(self.conv2d_list)):
            # multi-scale transform
            out = self.conv2d_list[it](x)
            d = out.size(1)

            # regional pooling
            out_diff = out #.reshape(bs, d, w * h).bmm(diffW).reshape(bs, d, w, h)
            #out_diff_all[it, :, :, :, :] = out_diff
            #out_diff_all.append(out_diff)

            # prediction
            out = self.conv1_list[it](out_diff)
            out = out.reshape(out.size(0), out.size(1), -1)  # out: bs, w*h, ncls

            pred = pred + out / nit

            # scaling before softmax
        #out = self.sofMax_list[0](out) #self.weight *
        #pred = self.sofMax_list[0](wei_cum)  # self.weight *


        #for it in range(len(self.conv2d_list)):
            #out = self.conv2d_list[it](x)
            #out_diff = out.reshape(bs, d, w * h).bmm(diffW).reshape(bs, d, w, h)

        #    out, wei = self.weiPooling_list[it](wei_cum, out_diff_all[it])

            # prediction
         #   out = self.pred_list[it](out, self.conv1_list[it].weight, self.conv1_list[it].bias)

            # transform to prob.
         #   bs, ncls = out.size()
         #   prob += (out) / nit #.reshape(bs, -1, ncls) / nit  # self.weight *
            #prob += self.sig2_list[it](out).reshape(bs, -1, ncls) / nit #self.weight *
        #prob = self.sig2_list[0](prob).reshape(bs, -1, ncls)  # self.weight *

        #if (prob != prob).sum():
        #    print(prob)

        return pred.reshape(bs,-1,w,h) #prob, wei_cum, wei # wei_cum

class SelfAttention_Module_MS(nn.Module):

    def __init__(self, n_stride, num_classes, dilation_series, padding_series):
        super(SelfAttention_Module_MS, self).__init__()

        self.ncls = num_classes
        self.conv2d_list = nn.ModuleList()
        self.conv1_list = nn.ModuleList()
        self.sig1_list = nn.ModuleList()
        self.sig2_list = nn.ModuleList()
        self.weiPooling_list = nn.ModuleList()
        self.pred_list = nn.ModuleList()
        self.sofMax_list = nn.ModuleList()

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))
            self.conv1_list.append(nn.Conv2d(1024, num_classes, kernel_size=1, stride=n_stride, bias=True))  # change
            self.sig1_list.append(nn.Sigmoid())
            self.sig2_list.append(nn.Sigmoid())
            self.sofMax_list.append(nn.Softmax(dim=1))
            self.weiPooling_list.append(weiPooling(num_classes))
            self.pred_list.append(pred_sw())

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

        for m in self.conv1_list:
            m.weight.data.normal_(0, 0.01)

        #self.sig1 = nn.Sigmoid()
        #self.sig2 = nn.Sigmoid()
        self.weiPooling = weiPooling(num_classes)
        #self.pred = pred_sw()
        #self.softmax2 = nn.Softmax(dim=1)

    def forward(self, x, diffW):

        bs, d, w, h = x.size()
        dim = 1024
        nit = len(self.conv2d_list)
        out_diff_all = [] #torch.zeros(nit, bs, dim, w, h).cuda()
        prob = torch.zeros(bs, self.ncls).cuda() #1,
        wei_cum = torch.zeros(bs, self.ncls, w * h).cuda()

        diffW = (diffW > 0.8).float() * diffW
        diffW_norm = torch.zeros_like(diffW)
        for l in range(x.size(0)):
            diffW_norm[l, :, :] = (1 / diffW[l, :, :].sum(1)).diag().mm(diffW[l, :, :]).permute(1, 0)

        for it in range(len(self.conv2d_list)):
            # multi-scale transform
            out = self.conv2d_list[it](x)
            d = out.size(1)

            out_diff_all.append(out)

            # regional pooling
            out_diff = out.reshape(bs, d, w * h).bmm(diffW_norm).reshape(bs, d, w, h)
            #out_diff_all[it, :, :, :, :] = out_diff
            #out_diff_all.append(out_diff)

            # prediction
            out = self.conv1_list[it](out_diff)
            out = out.reshape(out.size(0), out.size(1), -1)  # out: bs, w*h, ncls

            wei_cum = wei_cum + out / nit

            # scaling before softmax
        #out = self.sofMax_list[0](out) #self.weight *
        wei_cum = self.sofMax_list[0](wei_cum)  # self.weight *


        for it in range(len(self.conv2d_list)):
            #out = self.conv2d_list[it](x)
            #out_diff = out.reshape(bs, d, w * h).bmm(diffW).reshape(bs, d, w, h)

            out, wei = self.weiPooling_list[it](wei_cum, out_diff_all[it])

            # prediction
            out = self.pred_list[it](out, self.conv1_list[it].weight, self.conv1_list[it].bias)

            # transform to prob.
            bs, ncls = out.size()
            prob += (out) / nit #.reshape(bs, -1, ncls) / nit  # self.weight *
            #prob += self.sig2_list[it](out).reshape(bs, -1, ncls) / nit #self.weight *
        prob = self.sig2_list[0](prob).reshape(bs, -1, ncls)  # self.weight *

        if (prob != prob).sum():
            print(prob)

        return prob, wei_cum, wei # wei_cum

class SelfAttention_Module_multiscale_v2(nn.Module):

    def __init__(self, n_stride, num_classes, dilation_series, padding_series):
        super(SelfAttention_Module_multiscale_v2, self).__init__()

        self.ncls = num_classes
        self.conv2d_list = nn.ModuleList()
        self.conv1_list = nn.ModuleList()
        self.sig1_list = nn.ModuleList()
        self.sig2_list = nn.ModuleList()
        self.weiPooling_list = nn.ModuleList()
        self.pred_list = nn.ModuleList()
        self.sofMax_list = nn.ModuleList()

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))
            self.conv1_list.append(nn.Conv2d(1024, num_classes, kernel_size=1, stride=n_stride, bias=True))  # change
            self.sig1_list.append(nn.Sigmoid())
            self.sig2_list.append(nn.Sigmoid())
            self.sofMax_list.append(nn.Softmax(dim=1))
            self.weiPooling_list.append(weiPooling_sm(num_classes))
            self.pred_list.append(pred_sw())

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

        for m in self.conv1_list:
            m.weight.data.normal_(0, 0.01)

        #self.sig1 = nn.Sigmoid()
        #self.sig2 = nn.Sigmoid()
        self.weiPooling = weiPooling(num_classes)
        #self.pred = pred_sw()
        #self.softmax2 = nn.Softmax(dim=1)
        self.multi_scale_list = [0.1, 1, 5, 10, 20, 50,100] #, 5, 10, 20, 50] #, 100]

    def forward(self, x, diffW):

        bs, d, w, h = x.size()
        dim = 1024
        nit = len(self.conv2d_list)
        nsc = len(self.multi_scale_list)
        out_diff_all = torch.zeros(bs, nsc, dim, w * h).cuda()
        prob = torch.zeros(bs, self.ncls).cuda() #1,
        wei_cum = torch.zeros(bs, nsc, self.ncls, w * h).cuda()

        out = self.conv2d_list[0](x)
        d = out.size(1)
        out_feat = out
        for it in range(nsc):
            # multi-scale transform
            sc = self.multi_scale_list[it]
            diffW_cs = diffW.pow(sc)
            diffW_cs = (diffW_cs > 0.8).float() * diffW_cs
            diffW_norm = torch.zeros_like(diffW_cs)

            for l in range(x.size(0)):
                diffW_norm[l, :, :] = (1 / diffW_cs[l, :, :].sum(1)).diag().mm(diffW_cs[l, :, :]).permute(1, 0)

            # regional pooling
            out_diff = out_feat.reshape(bs, d, w * h).bmm(diffW_norm).reshape(bs, d, w, h)
            out_diff_all[:, it, :, :] = out_diff.reshape(bs, -1, w * h)

            # prediction
            out = self.conv1_list[0](out_diff)
            out = out.reshape(out.size(0), out.size(1), -1)
            wei_cum[:, it, :, :] = out

            # scaling before softmax
        with torch.no_grad():
            scale = wei_cum.argmax(1)
            wei_mask = torch.zeros(wei_cum.size()).cuda()
            wei_mask.scatter_(1, scale.unsqueeze(1), 1)

        wei_cum = self.sofMax_list[0]((wei_cum * wei_mask).sum(1))  # self.weight *
        wei_pool =  self.weiPooling_list[0](wei_cum)   # compute the pooling weights as softmax

        for it in range(self.ncls): #nsc
            # prepare the feature at approporiate scale
            with torch.no_grad():
                wei_mask_cls = torch.zeros(out_diff_all.size()).cuda()
                dim = out_diff_all.size(2)
                wei_mask_cls.scatter_(1, scale[:, it, :].unsqueeze(1).unsqueeze(1).repeat(1,1,dim,1), 1)

            feat_cls = (out_diff_all * wei_mask_cls).sum(1)

            # pooling the feature to be a single feature
            feat_cls_sg = feat_cls.bmm(wei_pool[:, it, :].unsqueeze(2)).squeeze()

            # prediction
            prob[:, it] = self.sig2_list[0]((feat_cls_sg * self.conv1_list[0].weight[it, :, :, :].squeeze().repeat(bs, 1) + self.conv1_list[0].bias[it].repeat(bs, 1)).sum(1))


        if (prob != prob).sum():
            print(prob)

        return prob, wei_cum, wei_pool, scale # wei_cum

class SelfAttention_Module_multiscale(nn.Module):

    def __init__(self, n_stride, num_classes, dilation_series, padding_series):
        super(SelfAttention_Module_multiscale, self).__init__()

        self.ncls = num_classes
        self.conv2d_list = nn.ModuleList()
        self.conv1_list = nn.ModuleList()
        self.sig1_list = nn.ModuleList()
        self.sig2_list = nn.ModuleList()
        self.weiPooling_list = nn.ModuleList()
        self.pred_list = nn.ModuleList()
        self.sofMax_list = nn.ModuleList()
        self.softMax_scale = softMax_scale(num_classes)

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))
            self.conv1_list.append(nn.Conv2d(1024, num_classes, kernel_size=1, stride=n_stride, bias=True))  # change
            self.sig1_list.append(nn.Sigmoid())
            self.sig2_list.append(nn.Sigmoid())
            self.sofMax_list.append(nn.Softmax(dim=1))
            self.weiPooling_list.append(weiPooling(num_classes))
            self.pred_list.append(pred_sw())

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

        for m in self.conv1_list:
            m.weight.data.normal_(0, 0.01)

        #self.sig1 = nn.Sigmoid()
        #self.sig2 = nn.Sigmoid()
        self.weiPooling = weiPooling(num_classes)
        #self.pred = pred_sw()
        #self.softmax2 = nn.Softmax(dim=1)
        self.multi_scale_list = nn.Parameter(torch.tensor([0.95, 0.75, 0.6, 0.4])) #[0.97, 0.75, 0.6, 0.4] #[1.0, 2.0, 5.0, 15.0, 40.0] #nn.Parameter(torch.tensor[1, 2, 5, 15])
        #[0.95, 0.75, 0.6, 0.4] # nn.Parameter nn.Parameter(torch.tensor([0.99, 0.9, 0.7, 0.5, 0.3, 0.1])) # [0.05, 0.2, 1, 5, 10] #  [0.05, 0.2, 1, 5, 10, 20, 50, 100] #0.05, 0.2, [0.05, 0.2,  1, 5, 10, 20, 50]  #[0.05, 0.2,  1, 5, 10, 20, 50] #, 5, 10, 20, 50] #, 100]
        self.featUpsample = torch.nn.Upsample(size=(41, 41), mode='bilinear')

    def forward(self, x, diffW):
        x = self.featUpsample(x)  # re-sampling to fixed size

        bs, d, w, h = x.size()
        dim = 1024
        nit = len(self.conv2d_list)
        nsc = len(self.multi_scale_list)
        out_diff_all = [] #torch.zeros(nit, bs, dim, w, h).cuda()
        prob = torch.zeros(bs, self.ncls).cuda() #1,
        wei_cum = torch.zeros(bs, nsc, self.ncls, w * h).cuda()

        out = self.conv2d_list[0](x)
        d = out.size(1)
        out_feat = out
        for it in range(nsc):
            # multi-scale transform
            sc = self.multi_scale_list[it]
            diffW_cs = diffW #.pow(sc)
            diffW_cs = (diffW_cs > sc).float() # * diffW_cs
            diffW_norm = torch.zeros_like(diffW_cs)

            for l in range(x.size(0)):
                diffW_norm[l, :, :] = (1 / diffW_cs[l, :, :].sum(1)).diag().mm(diffW_cs[l, :, :]).permute(1, 0)

            # regional pooling
            out_diff = out_feat.reshape(bs, d, w * h).bmm(diffW_norm).reshape(bs, d, w, h)
            out_diff_all.append(out_diff)

            # prediction
            out = self.conv1_list[0](out_diff)
            out = out.reshape(out.size(0), out.size(1), -1)
            wei_cum[:, it, :, :] = out

            # scaling before softmax
        #with torch.no_grad():
            #scale = wei_cum.argmax(1)
        wei_mask = self.softMax_scale(wei_cum)

        #wei_mask = torch.zeros(wei_cum.size()).cuda()
        #wei_mask.scatter_(1, scale.unsqueeze(1), 1)

        wei_cum = self.sofMax_list[0]((wei_cum * wei_mask).sum(1))  # self.weight *

        #for it in range(nsc): #nsc
        #    out, wei = self.weiPooling_list[it](wei_cum, out_diff_all[it])

            # prediction
         #   out = self.pred_list[0](out, self.conv1_list[0].weight, self.conv1_list[0].bias)

            # transform to prob.
        #    bs, ncls = out.size()
        #    prob += out * wei_mask[:, it, :, :]#/ nit #.reshape(bs, -1, ncls) / nit  # self.weight *
                #prob += self.sig2_list[it](out).reshape(bs, -1, ncls) / nit #self.weight *
        #prob = self.sig2_list[0](prob).reshape(bs, -1, ncls)  # self.weight *

        out, wei = self.weiPooling_list[0](wei_cum, out_feat)

        out = self.pred_list[0](out, self.conv1_list[0].weight, self.conv1_list[0].bias)
        bs, ncls = out.size()
        prob = self.sig2_list[0](out).reshape(bs, -1, ncls)

        if (prob != prob).sum():
            print(prob)

        return prob, wei_cum, wei, wei_mask # wei_cum

class SelfAttention_Module_norm(nn.Module):

    def __init__(self, n_stride, num_classes):
        super(SelfAttention_Module_v2, self).__init__()
        self.conv1 = nn.Conv2d(2048, num_classes, kernel_size=1, stride=n_stride, bias=True)  # change
        self.sig1 = nn.Sigmoid()

        #self.conv2 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=n_stride, bias=False)
        #self.softmax1 = nn.Softmax(dim=2)
        #self.conv3 = nn.Conv2d(2048, 1, kernel_size=1, stride=n_stride, bias=False)

        self.sig2 = nn.Sigmoid()
        self.weiPooling = weiPooling(num_classes)

        self.pred = pred_sw()
        self.softmax2 = nn.Softmax(dim=1)

        self.conv1.weight.data.normal_(0, 0.01)
        #nn.init.constant_(self.conv2.weight.data, 0)

        #self.conv3.weight.data.normal_(0, 0.01)
        #self.conv2.weight.data.normal_(0, 0.01)

        self.weight = nn.Parameter(torch.tensor([5.0]))
        #self.weights2 = nn.Parameter(torch.ones(num_classes, 2048) * 0.01).cuda()

    def forward(self, x, diffW):
        bs, d, w, h = x.size()
        out_diff = x.reshape(bs,d,w*h).bmm(diffW).reshape(bs, d, w, h)

        # feature normalization
        norm = torch.clamp(out_diff.norm(p=2, dim=1, keepdim=True), min = 1e-5)
        out_diff_norm = out_diff.div(norm.expand_as(out_diff))

        # weight normalization
        wnorm = torch.clamp(self.conv1.weight.norm(p=2, dim=1, keepdim=True), min = 1e-5)
        #wei_norm = self.conv1.weight.div(wnorm.expand_as(self.conv1.weight))
        #self.conv1.weight = wei_norm

        # prediction
        out = self.conv1(out_diff_norm)
        out = out.div(wnorm.permute(1,0,2,3).expand_as(out))
        out = out.reshape(out.size(0), out.size(1), -1)                         # out: bs, w*h, ncls

        # scaling before softmax
        out = self.softmax2(self.weight * out)
        out, wei = self.weiPooling(out, out_diff)

        # prediction
        out = self.pred(out, self.conv1.weight, self.conv1.bias)
        out = out.div(wnorm.squeeze().expand_as(out))

        # transform to prob.
        bs, ncls = out.size()
        out = self.sig2(self.weight * out).reshape(bs, -1, ncls)

        if (out != out).sum():
            print(out)

        return out, wei


class SelfAttention_Module(nn.Module):

    def __init__(self, n_stride, num_classes):
        super(SelfAttention_Module, self).__init__()
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=n_stride, bias=False)  # change
        self.sig1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=n_stride, bias=False)
        self.softmax1 = nn.Softmax(dim=2)
        #self.conv3 = nn.Conv2d(2048, 1, kernel_size=1, stride=n_stride, bias=False)
        self.sig2 = nn.Sigmoid()
        self.weiPooling = weiPooling(num_classes)
        self.pred = pred(num_classes)
        self.softmax2 = nn.Softmax(dim=1)

        self.conv1.weight.data.normal_(0, 0.01)
        nn.init.constant_(self.conv2.weight.data, 0)
        #self.conv3.weight.data.normal_(0, 0.01)
        #self.conv2.weight.data.normal_(0, 0.01)

        #self.weight1 = nn.Parameter(torch.tensor([1e+2])).cuda()
        #self.weights2 = nn.Parameter(torch.ones(num_classes, 2048) * 0.01).cuda()

    def forward(self, x, diffW):
        bs, d, w, h = x.size()
        out_diff = x.reshape(bs,d,w*h).bmm(diffW).reshape(bs, d, w, h)
        out = self.conv1(out_diff)                                               # x  : bs, w, h, 1024
        out = self.sig1(out)                                                # out: bs, w, h, 1024
        out = self.conv2(out)                                               # out: bs, w, h, ncls (21)
        out = out.reshape(out.size(0), out.size(1), -1)                     # out: bs, w*h, ncls

        '''
        out = self.softmax1(out * self.weight1)                             # out: bs, w*h, ncls (softmax across w*h)
        out = out.bmm(x.reshape(bs, d, w*h).permute(0, 2, 1))               # out: bs, ncls, 2048
        out = (out * self.weights2.repeat(bs, 1, 1)).sum(2)
        '''
        #'''
        #wei_tmp = out

        out = self.softmax2(out)
        wei_tmp = out

        out, wei = self.weiPooling(out, out_diff)
        #'''

        out = self.pred(out)

        bs, ncls = out.size()
        out = self.sig2(out).reshape(bs, -1, ncls)
        #out = torch.cat((out, 1-out), 1)

        return out, wei

class weiPooling_avr(nn.Module):
    def __init__(self):
        super(weiPooling_avr, self).__init__()
        #self.weights = nn.Parameter(torch.tensor([1.0]))
        # self.weight2 = nn.Parameter(torch.tensor([4]))

        #self.softmax1 = nn.Softmax(dim=2)
        #self.softmax2 = nn.Softmax(dim=1)

    def forward(self, input, x):
        bs, d1, n = input.size()
        bs,d,w,h=x.size()
        #wei = self.softmax1(input * torch.exp(self.weights[0]))  # out: bs, w*h, ncls (softmax across w*h)
        # wei = self.softmax2(wei * torch.exp(self.weights[1]))

        wei = input / (input.sum(2).reshape(bs, d1, 1).repeat(1,1,n))

        out = wei.bmm(x.reshape(bs, d, w * h).permute(0, 2, 1))  # out: bs, ncls, 2048
        return out, wei

class weiPooling(nn.Module):
    def __init__(self, ncls):
        super(weiPooling, self).__init__()
        self.weights = nn.Parameter(torch.ones(1, ncls) * 5.0) #1.6
        #self.weight2 = nn.Parameter(torch.tensor([4]))

        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, input, x):
        bs, d, w, h = x.size()
        ncls = input.size(1)
        df = input.size(2)

        #self.weights = nn.Parameter(torch.ones(1, ncls) * 1.5).cuda()  # 1.6
        input = self.softmax2(input) # semantic segmentation results
        wei = self.softmax1(input * torch.exp(self.weights[0]).reshape(1, ncls, 1).repeat(bs, 1, df))  # out: bs, w*h, ncls (softmax across w*h)

        #for l in range(ncls):
        #    wei[:,l,:] = self.softmax1(input[:,l,:] * torch.exp(self.weights[ncls]))
        #wei = self.softmax2(wei * torch.exp(self.weights[1]))

        out = wei.bmm(x.reshape(bs, d, w * h).permute(0, 2, 1))  # out: bs, ncls, 2048
        return out, input #wei

class softMax_scale(nn.Module):
    def __init__(self, ncls):
        super(softMax_scale, self).__init__()
        self.weights = nn.Parameter(torch.ones(1, ncls) * 1.5)  # 1.6
        self.sm = nn.Softmax(dim=1)

    def forward(self, input):
        bs = input.size(0)
        ns = input.size(1)
        ncls = input.size(2)
        d = input.size(3)
        out = self.sm(self.weights.unsqueeze(0).unsqueeze(3).repeat(bs, ns, 1, d) * input)

        return out

class weiPooling_sm(nn.Module):
    def __init__(self, ncls):
        super(weiPooling_sm, self).__init__()
        self.weights = nn.Parameter(torch.ones(1, ncls) * 1.6) #1.6
        #self.weight2 = nn.Parameter(torch.tensor([4]))

        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, input):
        bs = input.size(0)
        ncls = input.size(1)
        df = input.size(2)

        #self.weights = nn.Parameter(torch.ones(1, ncls) * 1.5).cuda()  # 1.6

        wei = self.softmax1(input * torch.exp(self.weights[0]).reshape(1, ncls, 1).repeat(bs, 1, df))  # out: bs, w*h, ncls (softmax across w*h)

        #for l in range(ncls):
        #    wei[:,l,:] = self.softmax1(input[:,l,:] * torch.exp(self.weights[ncls]))
        #wei = self.softmax2(wei * torch.exp(self.weights[1]))

        #out = wei.bmm(x.reshape(bs, d, w * h).permute(0, 2, 1))  # out: bs, ncls, 2048
        return wei

class pred(nn.Module):
    def __init__(self, ncls):
        super(pred, self).__init__()
        self.weights = nn.Parameter(torch.randn(ncls, 2048) * 0.01)
        self.bias = nn.Parameter(torch.zeros(ncls))

    def forward(self, input):
        bs = input.size(0)
        out = (input * self.weights.repeat(bs, 1, 1)).sum(2) + self.bias.repeat(bs, 1)
        return out

class pred_sw(nn.Module):
    def __init__(self):
        super(pred_sw, self).__init__()
        #self.weights = nn.Parameter(torch.randn(ncls, 2048) * 0.01)
        #self.bias = nn.Parameter(torch.zeros(ncls))

    def forward(self, input, weights, bias):
        bs = input.size(0)

        # feature normalization
        # norm = torch.clamp(input.norm(p=2, dim=2, keepdim=True), min=1e-5)
        # input_norm = input.div(norm.expand_as(input))

        out = (input * weights.squeeze().repeat(bs, 1, 1)).sum(2) + bias.repeat(bs, 1)
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

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par, momentum=0.8)
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
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par, momentum=0.8))
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

    def print_grad(self, grad_input, grad_output):
        print(torch.isnan(grad_input.view(-1)).sum())
        print(torch.isnan(grad_output.view(-1)).sum())


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x.register_hook(print)

        x = self.layer5(x)

        #x.register_hook(print)
        return x

class ResNet_atten(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet_atten, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par, momentum=0.8)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_atten_layer(SelfAttention_Module_multiscale, 1, num_classes, [1],[1]) #, 2, 3_v, 2, 3 2  SelfAttention_Module_MS
        #self.layer5 = self._make_atten_layer(SelfAttention_Module_MS, 1, num_classes, [1, 3, 5],[1, 3, 5])  # , 2, 3_v, 2, 3 2  SelfAttention_Module_MS

        #self.layer5 = self._make_atten_layer(SelfAttention_Module_v2, 1, num_classes, [1], [1])  # _v2  SelfAttention_Module_MS
        self.featUpsample = torch.nn.Upsample(size=(41, 41), mode='bilinear')
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False
        '''

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par, momentum=0.8))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_atten_layer(self,block, stride, num_classes, dilation_series, padding_series):
        return block(stride,num_classes,dilation_series, padding_series)

    def print_grad(self, grad_input, grad_output):
        print(torch.isnan(grad_input.view(-1)).sum())
        print(torch.isnan(grad_output.view(-1)).sum())

    def forward(self, inputs):
        (x, diffW) = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x.register_hook(print)

        #x, wei_cum, wei, scale = self.layer5(x, diffW) #self.featUpsample(x)
        x, wei_cum, wei, scale = self.layer5(x, diffW)  # self.featUpsample(x)

        #x.register_hook(print)
        return x, wei_cum, wei, scale

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

def Res_Ms_Deeplab(num_classes=21):
    model = MS_Deeplab(Bottleneck, num_classes)
    return model

#def Res_Deeplab(num_classes=21, is_refine=False):
#    if is_refine:
#        model = ResNet_Refine(Bottleneck,[3, 4, 23, 3], num_classes)
#    else:
#        model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
#    return model

def Res_Deeplab(num_classes=21, isatten=False):
    if isatten:
        model = ResNet_atten(Bottleneck,[3, 4, 23, 3], num_classes)
    else:
        model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model


'''
#print(out.argmax(2))
import matplotlib.pyplot as plt
plt.imshow(inputs[0][0,0,:,:])
plt.show()
plt.imshow(diffW[0,:,756].reshape(41,41))
plt.show()
'''