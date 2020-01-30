import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import SpixelAggr_avr_cuda
import numpy as np
import time
from diffMap_deeplab.diffMap_functional import  *

# Feat2Dist
class Feat2Dist(nn.Module):
    def __init__(self):
        super(Feat2Dist, self).__init__()

    def forward(self, input):
        return Feat2Dist_function.apply(input)

# Feat2Dist
class Feat2Dist_batch(nn.Module):
    def __init__(self):
        super(Feat2Dist_batch, self).__init__()

    def forward(self, input):
        return Feat2Dist_function_batch.apply(input)

# super-pixel aggragation
class SpixelAggr_avr(nn.Module):
    def __init__(self):
        super(SpixelAggr_avr, self).__init__()
        #self.input_features = input_feature
        #self.segLabels = segLabels
        #self.coor_idx = coor_idx

    def forward(self, input, segLabels, coor_idx):
        #(input, segLabels, coor_idx) = inputs
        return SpixelsAggr_avr_function.apply(input, segLabels, coor_idx)


# reshape to connect deeplab and feat2dist
class TensorReshape(nn.Module):
    def __init__(self):
        super(TensorReshape, self).__init__()
        #self.input_features = input_feature
        #self.segLabels = segLabels
        #self.coor_idx = coor_idx

    def forward(self, input, segLabels, coor_idx):
        return input


# super-pixel aggragation
class SpixelAggr_avr_dense(nn.Module):
    def __init__(self):
        super(SpixelAggr_avr_dense, self).__init__()
        #self.input_features = input_feature
        #self.segLabels = segLabels
        #self.coor_idx = coor_idx

    def forward(self, input, segLabels):
        #(input, segLabels, coor_idx) = inputs
        return SpixelsAggr_avr_dense_function.apply(input, segLabels)

# dist2SimiMatrix
class dist2SimiMatrix(nn.Module):
    def __init__(self, wei):
        super(dist2SimiMatrix, self).__init__()
        self.weights = nn.Parameter(torch.log(torch.Tensor(([wei]))).double())

    def forward(self, input):
        return dist2SimiMatrix_function.apply(input, self.weights)

# dist2SimiMatrix
class dist2SimiMatrix_batch(nn.Module):
    def __init__(self, wei):
        super(dist2SimiMatrix_batch, self).__init__()
        self.weights = nn.Parameter(torch.log(torch.Tensor(([wei]))).double())

    def forward(self, input):
        return dist2SimiMatrix_function_batch.apply(input, self.weights)


# dist2SimiMatrix_scale
class dist2SimiMatrix_scale(nn.Module):
    def __init__(self):
        super(dist2SimiMatrix_scale, self).__init__()
        self.weights = nn.Parameter(torch.log(torch.Tensor([1e-3])).double())


    def forward(self, input):
        return dist2SimiMatrix_scale_function.apply(input, self.weights)


# neighMasking
class neighMasking(nn.Module):
    def __init__(self):
        super(neighMasking, self).__init__()
        #self.state_size = state_size
        #self.neigMask = neigMask

    def forward(self, input, neigMask):
        return neighMasking_function.apply(input, neigMask)

# neighMasking
class neighMasking_batch(nn.Module):
    def __init__(self):
        super(neighMasking_batch, self).__init__()
        #self.state_size = state_size
        #self.neigMask = neigMask

    def forward(self, input, neigMask):
        return neighMasking_function_batch.apply(input, neigMask)

#compNormSimiMatrix
class compNormSimiMatrix(nn.Module):
    def __init__(self):
        super(compNormSimiMatrix, self).__init__()

    def forward(self, input):
        return compNormSimiMatrix_function.apply(input)


#compNormSimiMatrix
class compNormSimiMatrix_batch(nn.Module):
    def __init__(self):
        super(compNormSimiMatrix_batch, self).__init__()

    def forward(self, input):
        return compNormSimiMatrix_function_batch.apply(input)


#compNormSimiMatrix
class compNormSimiMatrix_batch(nn.Module):
    def __init__(self):
        super(compNormSimiMatrix_batch, self).__init__()

    def forward(self, input):
        return compNormSimiMatrix_function_batch.apply(input)


# compEigDecomp
class compEigDecomp(nn.Module):
    def __init__(self):
        super(compEigDecomp, self).__init__()

    def forward(self, input):
        return compEigDecomp_function.apply(input)


# compEigDecomp
class compEigDecomp_batch(nn.Module):
    def __init__(self):
        super(compEigDecomp_batch, self).__init__()

    def forward(self, input):
        # power method
        [bs, w, w] = input.size()
        ns = 50
        x = torch.eye(w, ns).cuda().double().repeat(bs, 1, 1)
        t = torch.zeros(ns, ns).repeat(bs, 1, 1).double().cuda()

        self.qr = compQRDecomp()

        # power iteration
        for p in range(2):
            for i in range(5):
                x = input.bmm(x)
                x = F.normalize(x, p=2, dim=1)
            x, t = self.qr(x)

        t0 = torch.zeros(bs, ns).double().cuda()
        for l in range(bs):
            t0[l, :] = torch.abs(t[l, :, :].diag())

        return t0, x


# QR decomp.
class compQRDecomp(nn.Module):
    def __init__(self):
        super(compQRDecomp, self).__init__()

    def forward(self, input):
        return  compQRDecomp_function.apply(input)


# compDiffDist
class compDiffDist(nn.Module):
    def __init__(self):
        super(compDiffDist, self).__init__()
        self.weights = nn.Parameter(torch.Tensor([8]).double())

    def forward(self, input_G, input_V):
        return compDiffDist_function.apply(input_G, input_V, self.weights)

# compDiffDist
class compDiffDist_batch(nn.Module):
    def __init__(self):
        super(compDiffDist_batch, self).__init__()
        self.weights = nn.Parameter(torch.Tensor([8]).double())

    def forward(self, G, V):
        t = self.weights

        bs, r, d = V.size()
        G = torch.max(G, torch.zeros_like(G))
        E = torch.ones(bs, r, d, dtype=torch.float64).cuda()
        eVs = torch.ones(bs, d, d, dtype=torch.float64).cuda()

        Phi = V
        Phi2 = V.pow(2)

        for l in range(bs):
            eVs[l, :, :] = G[l, :].pow(2 * t).diag()

        y = E.bmm(eVs).bmm(Phi2.permute(0, 2, 1)) + Phi2.bmm(eVs).bmm(E.permute(0, 2, 1)) - 2 * Phi.bmm(eVs).bmm(Phi.permute(0, 2, 1))

        return y
        #return compDiffDist_function_batch.apply(input_G, input_V, self.weights)

# compLoss
class loss_kernelMatching(nn.Module):
    def __init__(self):
        super(loss_kernelMatching, self).__init__()

    def forward(self, pred, targ):
        return kernelMatching_function.apply(pred, targ)


# compLoss
class loss_kernelMatching_batch(nn.Module):
    def __init__(self, target):
        super(loss_kernelMatching_batch, self).__init__()
        self.target = target.double()

    def forward(self, pred):
        a = time.time()

        bs = pred.size(0)
        losses = torch.zeros(bs, 1).cuda().double()

        target = self.target.double()

        for l in range(bs):
            no_pr = torch.norm(pred[l, :, :], 2)
            no_gt = torch.norm(target[l, :, :], 2)

            losses[l] = - (pred[l, :, :] * target[l, :, :]).sum() / (no_gt * no_pr)
        #'''

        b = time.time()

        print(b-a)
        return (losses.sum() / bs)