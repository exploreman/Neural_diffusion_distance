import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import SpixelAggr_avr_cuda
import numpy as np


# Feat2Dist
class Feat2Dist_function(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        imgx = input.squeeze()
        r, c = imgx.size()
        xnorm = imgx.power(2).sum(1).reshape(r, 1)
        ONE = torch.ones(r, 1)
        y = (ONE.mul(xnorm) + xnorm.mul(ONE.transpose(1,0))) - 2 * imgx.mul(imgx.transpose(0,1))
        return y

    @staticmethod
    def backward(self, dzdy):
        input = self.saved_variables
        imgx = input.squeeze()
        r, c = imgx.size()
        ONE = torch.ones(r,1)
        y = 2 * (dzdy.mul(ONE).diag() + ONE.transpose(1,0).mul(dzdy).diag() - dzdy - dzdy.transpose(1,0)).mul(imgx)
        return y


class Feat2Dist(nn.Module):
    def __init__(self, state_size):
        super(Feat2Dist, self).__init__()
        self.state_size = state_size

    def forward(self, input, state):
        return Feat2Dist_function.apply(input, *state)


# super-pixel aggragation
class SpixelsAggr_avr_function(Function):
    @staticmethod
    def forward(self, input, segLabels, coor_idx):
        a = np.ones(segLabels.size())
        nSegs = np.bincount(segLabels, weights = a)
        output = SpixelAggr_avr_cuda.forward(input, segLabels[coor_idx], nSegs)
        self.save_for_backward(input, segLabels, nSegs)

        return output

    @staticmethod
    def backward(self, grad_out):
        input, segLabels, nSegs = self.saved_variables
        grad_in = SpixelAggr_avr_cuda.backward(grad_out.contiguous(), input, segLabels, nSegs)

        return grad_in


#class SpixelAggr_avr(nn.Module):
#    def __init__(self, input_features, segLabels, coor_idx, state_size):
#        super(SpixelAggr_avr, self).__init__()
#        self.input_features = input_features
#        self.segLabels = segLabels
#        self.coor_idx = coor_idx

#    def forward(self, input, state):
#        return SpixelsAggr_avr_function.apply(input, self.segLabels, self.coor_idx, *state)


# dist2SimiMatrix_scale
class dist2SimiMatrix_scale_function(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        r,d = input.size()
        scale = self.weights
        x = F.relu(input)
        ONE = torch.ones(r,1)
        xv = torch.sqrt(1 / (x * ONE)).diag()
        vx = torch.sqrt(1 / (ONE.transpose(1, 0) * x)).diag()
        y = torch.exp(-scale * xv.mm(x).mm(vx) * r)
        return y

    @staticmethod
    def backward(self, dzdy):
        input = self.saved_variables
        r, d = input.size()
        scale = self.weights
        x = F.relu(input)
        ONE = torch.ones(r, 1)
        xv = torch.sqrt(1 / (x * ONE)).diag()
        vx = torch.sqrt(1 / (ONE.transpose(1, 0) * x)).diag()
        dzds = -scale * dzdy * r * dzdy
        y = -0.5 * (dzds * (xv.pow(3).mm(x).mm(vx)).mm(ONE).mm(ONE.transpose(0,1))) - 0.5 * ONE * ONE.transpose(1, 0).mm(dzds * (xv.mm(x).mm(vx.pow(3)))) + xv.mm(dzds.mm(vx))
        dw = (-dzdy * (xv.mm(x).mm(vx)) * r).sum()
        return y, dw


class dist2SimiMatrix_scale(nn.Module):
    def __init__(self, state_size):
        super(dist2SimiMatrix_scale).__init__()
        self.state_size = state_size
        self.weights = nn.Parameter(torch.Tensor(1))

    def forward(self, input, state):
        return dist2SimiMatrix_scale_function.apply(input, self.weights)


# neighMasking
class neighMasking_function(Function):
    @staticmethod
    def forward(self, input, neigW):
        neigMask = neigW.mm(neigW).mm(neigW).mm(neigW).mm(neigW).mm(neigW)
        y = input * (neigMask + neigMask.transpose(1,0) > 0)
        return y

    @staticmethod
    def backward(self, input, neigW, dzdy):
        neigMask = neigW.mm(neigW).mm(neigW).mm(neigW).mm(neigW).mm(neigW)
        y = dzdy * ((neigMask + neigMask.transpose(1,0)) > 0)
        return y


class neighMasking(nn.Module):
    def __init__(self, neigMask, state_size):
        super(neighMasking).__init__()
        self.state_size = state_size
        self.neigMask = neigMask

    def forward(self, input, state):
        return neighMasking_function.apply(input, self.neigMask)


#compNormSimiMatrix
class compNormSimiMatrix_function(Function):
    @staticmethod
    def forward(self, input):
        r,d = input.size()
        ONE = torch.ones(r, 1)
        simi = input.squeeze()
        EPS = torch.full((r,1), 1e-5)
        D = 1 / torch.max(simi.sum(1), EPS)
        y = D.diag() * simi
        return y

    @staticmethod
    def backward(self, input, dzdy):
        r, d = input.size()
        ONE = torch.ones(r, 1)
        simi = input.squeeze()
        EPS = torch.full((r, 1), 1e-5)
        D = 1 / torch.max(simi.sum(1), EPS)
        y = D.diag().mm(dzdy) - (D.pow(2).diag().mm(simi).mm(dzdy.transpose(1,0))).mm(ONE.transpose(1,0))
        return y


class compNormSimiMatrix(nn.Module):
    def __init__(self, state_size):
        super(compNormSimiMatrix).__init__()
        self.state_size = state_size

    def forward(self, input, state):
        return compNormSimiMatrix_function.apply(input)


# compEigDecomp
class compEigDecomp_function(Function):
    @staticmethod
    def forward(self, input):
        x = input.squeeze()
        r, d = x.size()
        G, V = torch.symeig(x, eigenvectors=True)
        tmp, order = torch.sort(G.diag(), 0, descending=True)
        G = G[order].diag()
        V = V[:,order]
        eig = [G, V]
        variables = eig
        self.save_for_backward(*variables)
        return eig

    @staticmethod
    def backward(self, input, dzdy):
        eig = self.saved_variables
        G = eig[0]
        V = eig[1]
        dzdG = dzdy[0]
        dzdV = dzdy[1]
        x = input.squeeze()
        r, d = x.size()
        n_neig = dzdy.size(1)
        vone = torch.ones(r,1)
        dzdV = torch.cat((dzdV, torch.ones(r, r - n_neig)), 1)
        dzdG = torch.cat((dzdG.transpose(1,0), torch.zeros(1, r-n_neig)), 1)
        diff = G.diag().pow(2).mm(vone.transpose(1,0))-(G.diag().pow(2).mm(vone.transpose(1,0)))
        K = 1 / diff
        K[torch.eye(K.size(0)) > 0] = 0
        K[torch.isnan(K)] = 0
        K[K == float("Inf")] = 0
        y = V.mm(dzdG.transpose(1,0).diag()).mm(V.transpose(1,0)) + V.mm(K.transpose(1,0)) * (V.transpose(1,0).mm(dzdV)).mm(V.transpose(1,0))
        return y


class compEigDecomp(nn.Module):
    def __init__(self, state_size):
        super(compEigDecomp).__init__()
        self.state_size = state_size

    def forward(self, input, state):
        return compEigDecomp_function.apply(input)


# compDiffDist
class compDiffDist_function(Function):
    @staticmethod
    def forward(self, input):
        self.s(input)
        eig = input
        G = eig[0]
        V = eig[1]
        r, d = V.size()
        n_eigs = 100
        t = self.scale
        st = 1
        eig[0] = torch.max(G, torch.zeros_like(G))
        E = torch.ones(r, n_eigs - st + 1)
        Phi = V[:, st-1: n_eigs]
        Phi2 = Phi.pow(2)
        eVs = ((G[st-1: n_eigs, st-1: n_eigs]).diag().pow(2 * t)).diag()
        y = E.mm(eVs).mm(Phi2.transpose(1,0)) + Phi2.mm(eVs).mm(E.transpose(1,0)) - 2 * Phi.mm(eVs).mm(Phi.transpose(1,0))
        return y

    @staticmethod
    def backward(self, dzdy):
        input = self.saved_variables
        G = input[0]
        V = input[1]
        r,d = V.size()
        n_eigs = 100
        vone = torch.ones(r, 1)
        t = self.scale
        st = 1
        dzdt = 0
        Phi = V[:, st: n_eigs]
        Gs = G
        y = 2 * (dzdy.transpose(1,0).mm(vone).diag()).mm(Phi) + (dzdy.mm(vone)).diag().mm(Phi) - dzdy.transpose(1,0).mm(Phi) - dzdy.mm(Phi).mm(G[st-1:n_eigs, st-1: n_eigs].pow(2 * t))
        Ks = torch.zeros(r * r, n_eigs - st + 1)
        for p in range(n_eigs - st + 1):
            tmp = vone.mm(Phi[:, p].pow(2)).transpose(1,0) +  Phi[:, p].pow(2).mm(vone.transpose(1,0)) - 2 * Phi[:,p].mu(Phi[:,p].transpose(1,0))
            Ks[:,p] = (2 * t * Gs[p, p].pow(2 * t - 1) * tmp).transpose()
            dzdt = dzdt + 2 * torch.log(max(Gs[p, p], 1e-10)) * Gs(p, p).pow(2 * t) * tmp
        vdzdy = dzdy.reshape(1, dzdy.size(1) * dzdy.size(0))
        z = vdzdy.mm(Ks)
        dzdt = vdzdy.mm(dzdt)
        z = z.transpose(1,0)
        return [z, y], dzdt


class compDiffDist(nn.Module):
    def __init__(self, state_size):
        super(compDiffDist).__init__()
        self.state_size = state_size

    def forward(self, input, state):
        return compDiffDist_function.apply(input)