import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import SpixelAggr_avr_cuda
import numpy as np
from math import sqrt
import time
from diffMap_deeplab.diffMap_functional import  *

'''
import matplotlib.pyplot as plt
plt.imshow(images[0,1,:,:])
plt.show()

plt.imshow(y[0,:,824].reshape(41,41).cpu().detach().numpy())
plt.show()
plt.imshow(y[0,:,:].cpu().detach().numpy())
plt.show()
'''

# CRF as RNN as post-process using diffusion distance based  affinity matrix
class CRF_RNN(nn.Module):
    def __init__(self, ncls):
        super(CRF_RNN, self).__init__()
        self.weights = nn.Parameter(torch.eye(ncls).cuda())

    def forward(self, seg, W):

        niters = 5
        #W = (W > 0.85).float() * W
        D = torch.zeros_like(W)
        #W1 = W
        softmax = torch.nn.Softmax(1)

        bs, d, r, c = seg.size()
        qVals =  seg.reshape(bs, d, r * c)
        uniqs = qVals

        #relu = torch.nn.ReLU()
        #eps = 1e-5
        # construct filtering matrix
        for i in range(bs):
            #W1[i, :, :] = W[i, :, :] - W[i, :, :].diag().diag()
            D[i, :, :] = (1 / W[i, :, :].sum(1)).diag().mm(W[i, :, :])
            # set matrix diagonal as zeros
            #norm = (W1[i, :, :].sum(1) - eps) + eps
            #D[i, :, :] = (1 / norm).diag().mm(W1[i, :, :])

        for id in range(niters):

            # normalization
            Q = softmax(qVals)

            # filtering the label confidence maps
            seg_diff = Q.bmm(D.permute(0, 2, 1))

            # compatability transform
            seg_update = self.weights.repeat(bs, 1, 1).bmm(seg_diff)

            # Adding Unary
            qVals = uniqs - seg_update


        return qVals.reshape(bs, d, r, c)

# Post-process of segmentation confidence map using affinity matrix
class Regu_diff(nn.Module):
    def __init__(self):
        super(Regu_diff, self).__init__()

    def forward(self, seg, W):
        # filtering the segmentation confidence maps
        relu = torch.nn.ReLU()
        bs = W.size(0)
        D = torch.zeros_like(W)
        W = (W > 0.85).float() * W #+ (W <= 0.7) * 0
        #W = relu(W)
        #W = W + 0.7
        #W_tmp[W_tmp < 0.7] = 0
        for i in range(bs):
            D[i, :, :] = (1/W[i,:,:].sum(1)).diag().mm(W[i, :, :])

        bs, d, r, c = seg.size()
        seg_diff = seg.reshape(bs, d, r * c).bmm(D.permute(0,2,1))
        return seg_diff.reshape(bs, d, r, c)


# Feat2Dist
class Feat2Dist_batch(nn.Module):
    def __init__(self):
        super(Feat2Dist_batch, self).__init__()

    def forward(self, input):
        bs, d, r, c = input.size()
        imgx = input.reshape(bs, d, r * c).permute(0, 2, 1)

        xnorm = imgx.pow(2).sum(2).reshape(bs, r * c, 1)
        vone = torch.ones(bs, r * c, 1).cuda()
        y = (vone.bmm(xnorm.permute(0, 2, 1)) + xnorm.bmm(vone.permute(0, 2, 1))) - 2 * imgx.bmm(imgx.permute(0, 2, 1))

        return y

# dist2SimiMatrix
class dist2SimiMatrix_batch(nn.Module):
    def __init__(self, wei):
        super(dist2SimiMatrix_batch, self).__init__()
        self.weights = nn.Parameter(torch.log(torch.Tensor(([wei]))))

    def forward(self, input):
        bs, r, d = input.size()
        scale = self.weights
        scale_exp = torch.exp(scale)
        x = F.relu(input)
        y = torch.exp(-scale_exp * x)

        #if torch.isnan(y).nonzero().__len__() > 0:
        #   y00000
        return y

# neighMasking
class neighMasking_batch(nn.Module):
    def __init__(self):
        super(neighMasking_batch, self).__init__()

    def forward(self, input, neigMask):
        #neigMask = Variable(neigMask, requires_grad = False)
        bs = input.size(0)
        y = input * neigMask.repeat(bs, 1, 1)
        return y


#compNormSimiMatrix
class compNormSimiMatrix_batch(nn.Module):
    def __init__(self):
        super(compNormSimiMatrix_batch, self).__init__()

    def forward(self, input):
        bs, r, d = input.size()
        simi = input
        EPS = torch.full((bs, r, 1), 1e-5).squeeze().cuda()
        D = 1 / torch.max(simi.sum(2), EPS)

        Ddiag = torch.zeros(bs, r, r).cuda()
        #Ddiag_sq = torch.zeros(bs, r, r).cuda()

        for l in range(bs):
            Ddiag[l, :, :] = D[l, :].diag()
            #Ddiag_sq[l, :, :] = D[l, :].pow(2).diag()

        y = Ddiag.bmm(simi)

        #if torch.isnan(y).nonzero().__len__() > 0:
        #    print(simi[0,:,:])
        #    y

        return y

# compEigDecomp
class compEigDecomp_batch(nn.Module):
    def __init__(self):
        super(compEigDecomp_batch, self).__init__()

    def forward(self, input, si):
        # power method
        [bs, w, w] = input.size()
        ns = 49
        ns_orig=ns
        r = si[0]
        d = si[1]

        wint = sqrt(w / ns)
        nW = round(r / wint)
        nH = round(ns / nW)
        wint = r / nW
        hint = d / nH
        ns = nW * nH
        x = torch.zeros(w, ns).cuda()

        id = 0
        for j in range(nW):
            for i in range(nH):
                pos = round((j + 0.5) * wint) * d + round((i + 0.5) * hint)
                x[pos, id] = 1
                id = id + 1

        #for l in range(ns):
        #    i = int(l / 7)
        #    j = int(l % 7)
        #    x[3 + (i * 6 + 3) * ws + 6 * j, l] = 1


        '''
        idx_po = (torch.rand(ns) * (w - 1)).int().unique()
        while idx_po.__len__() < ns:
            idx_ad = (torch.rand(ns - idx_po.__len__()) * (w - 1)).int().unique()
            idx_po = torch.cat([idx_ad, idx_po])
        idx = ((idx_po * ns).long() + torch.tensor(range(ns)))
        '''

        #idx = (((torch.tensor(list(range(ns))) * (w-1)/(ns-1)).int() * ns).long() + torch.tensor(range(ns)))
        #x.view(-1)[idx] = 1
        x = x.reshape(w,ns).repeat(bs, 1, 1)
        t = torch.zeros(ns, ns).repeat(bs, 1, 1).cuda()
        self.qr = compQRDecomp()

        iterOt = 2
        iterIn = 20

        # baseline (2, 20), Case 0: (1, 20); Case 1: (1, 10); Case 3: (2, 10), Case 4: (3, 20); Maximal iterations: 120000

        # power iteration
        if 1:
            #input_pow = input
            #for i in range(5):
            #    input_pow = input_pow.bmm(input)

            #input_pow = input_pow.bmm(input_pow)
            input = input.bmm(input)
            #input = input.pow(0.2)

            for p in range(iterOt):
                for i in range(iterIn): #12
                    x = input.bmm(x)
                    #x = F.normalize(x, p=2, dim=1)

                #if torch.isnan(x).nonzero().__len__() > 0:
                #    x
                #x = F.normalize(x, p=2, dim=1)
                x, t = self.qr(x)
        else:
            #input = input.bmm(input).bmm(input)

            for p in range(2):
                #for i in range(1):
                for i in range(20):
                    x = input.bmm(x)
                    x = F.normalize(x, p=2, dim=1)

                # if torch.isnan(x).nonzero().__len__() > 0:
                #    x

                x, t = self.qr(x)

        t0 = torch.zeros(bs, ns).cuda()
        for l in range(bs):
            t0[l, :] = torch.abs(t[l, :, :].diag())


        return t0, x


# QR decomp.
class compQRDecomp(nn.Module):
    def __init__(self):
        super(compQRDecomp, self).__init__()

    def forward(self, input):
        return compQRDecomp_function_f.apply(input)


# compDiffDist
class compDiffDist_batch(nn.Module):
    def __init__(self):
        super(compDiffDist_batch, self).__init__()
        self.weights = nn.Parameter(torch.Tensor([0.5])) #1.5

    def forward(self, G, V):
        t = self.weights

        bs, r, d = V.size()
        G = torch.max(G, torch.zeros_like(G))
        E = torch.ones(bs, r, d).cuda()
        eVs = torch.ones(bs, d, d).cuda()

        Phi = V
        Phi2 = V.pow(2)

        for l in range(bs):
            eVs[l, :, :] = G[l, :].pow(2 * t).diag()

        y = E.bmm(eVs).bmm(Phi2.permute(0, 2, 1)) + Phi2.bmm(eVs).bmm(E.permute(0, 2, 1)) - 2 * Phi.bmm(eVs).bmm(Phi.permute(0, 2, 1))



        return y
        #return compDiffDist_function_batch.apply(input_G, input_V, self.weights)

# interpolation
class inter_feat_linear(nn.Module):
    def __init__(self):
        super(inter_feat_linear, self).__init__()


    def forward(self, grid_inte, grid, feat, lbs, wei):
        # grid: N * gW * gH * 2 (N=1)
        # grid_inte: N * gW' * gH' * 2  (N=1)
        # feat: N * D * W * H
        # lbs: one hot code: N * D * W * NC  (NC = number of cluster)

        # the grid_inte, grid is defined over grid of "feat"
        r = grid.size(1)
        c = grid.size(2)
        N = feat.size(0)
        D = feat.size(1)

        ncls = lbs.size(1)

        r_in = grid_inte.size(1)
        c_in = grid_inte.size(2)

        grid_interval_X = grid[0,2,2,0] - grid[0,1,2,0]
        grid_interval_Y = grid[0,2,2,1] - grid[0,2,1,1]

        # step 1: fetch the feat for (grid, lbs) by linear interpolation
        #feat_grid = torch.nn.functional.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros')
        #if scale == 0:
        feat_upsmpl = torch.nn.Upsample(size = (r,c), mode = 'bilinear')
        #else:
        #    feat_upsmpl = feat

        feat_grid = feat_upsmpl(feat)

        with torch.no_grad():
            # step 2: compute the 4_NN for each location of grid_inte (4 or others?)
            grid_inte_rp = torch.zeros_like(grid_inte)
            grid_inte_rp[:,:,:,0] =  torch.floor(grid_inte[:,:,:,0]  / grid_interval_X)
            grid_inte_rp[:,:,:,1] =  torch.floor(grid_inte[:,:,:,1]  / grid_interval_Y)


        ### with grad:
        # step 3: compute voting weights for grid_inte voted from grid
        dist = torch.zeros(N, r_in * c_in).cuda()
        dist_sum = torch.zeros(N, r_in * c_in).cuda()
        lbs_voting = torch.zeros(N, ncls, r_in * c_in).cuda()
        feat_grid = feat_grid.reshape(N,D,-1)
        
        id = 0
        lbs = lbs.reshape(N, ncls, -1)
        for i in range(-2,3,1):
            for j in range(-2,3,1):

                # compute the voting weights
                grid_curr= torch.clamp(grid_inte_rp[:,:,:,0] + i, min=0, max=r-1) *c + torch.clamp(grid_inte_rp[:,:,:,1] + j, min=0, max=c-1)

                # get the feat
                feat_grid_curr = feat_grid[:, :, grid_curr.long()].squeeze(2)


                lbs_curr = lbs[:,:,grid_curr.long()].squeeze(2).reshape(N,ncls,-1)

                # voting weight
                dist = torch.exp(- (wei.cuda() * ((feat_grid_curr - feat).pow(2)).mean(1))).reshape(N,-1)

                lbs_voting = lbs_voting +  lbs_curr  *  dist.unsqueeze(1).repeat(1, ncls,1)

                dist_sum = dist_sum + dist

                id = id + 1


        # step 4: compute the interpolated label for grid_inte
        lbs_inte = lbs_voting / torch.clamp(dist_sum.unsqueeze(1).repeat(1,ncls,1), min=1e-15)

        return lbs_inte.reshape(N, ncls, r_in,c_in)


class inter_feat_linear_cpu(nn.Module):
    def __init__(self):
        super(inter_feat_linear_cpu, self).__init__()

    def forward(self, grid_inte, grid, feat, lbs, wei):
        # grid: N * gW * gH * 2 (N=1)
        # grid_inte: N * gW' * gH' * 2  (N=1)
        # feat: N * D * W * H
        # lbs: one hot code: N * D * W * NC  (NC = number of cluster)

        # the grid_inte, grid is defined over grid of "feat"
        r = grid.size(1)
        c = grid.size(2)
        N = feat.size(0)
        D = feat.size(1)

        ncls = lbs.size(1)

        r_in = grid_inte.size(1)
        c_in = grid_inte.size(2)

        grid_interval_X = grid[0, 2, 2, 0] - grid[0, 1, 2, 0]
        grid_interval_Y = grid[0, 2, 2, 1] - grid[0, 2, 1, 1]

        # step 1: fetch the feat for (grid, lbs) by linear interpolation
        # feat_grid = torch.nn.functional.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros')
        # if scale == 0:
        feat_upsmpl = torch.nn.Upsample(size=(r, c), mode='bilinear')
        # else:
        #    feat_upsmpl = feat

        feat_grid = feat_upsmpl(feat)

        with torch.no_grad():
            # step 2: compute the 4_NN for each location of grid_inte (4 or others?)
            grid_inte_rp = torch.zeros_like(grid_inte)
            grid_inte_rp[:, :, :, 0] = torch.floor(grid_inte[:, :, :, 0] / grid_interval_X)
            grid_inte_rp[:, :, :, 1] = torch.floor(grid_inte[:, :, :, 1] / grid_interval_Y)

        ### with grad:
        # step 3: compute voting weights for grid_inte voted from grid
        dist = torch.zeros(N, r_in * c_in)
        dist_sum = torch.zeros(N, r_in * c_in)
        lbs_voting = torch.zeros(N, ncls, r_in * c_in)
        feat_grid = feat_grid.reshape(N, D, -1)

        id = 0
        lbs = lbs.reshape(N, ncls, -1)
        for i in range(-2, 3, 1):
            for j in range(-2, 3, 1):
                # compute the voting weights
                grid_curr = torch.clamp(grid_inte_rp[:, :, :, 0] + i, min=0, max=r - 1) * c + torch.clamp(
                    grid_inte_rp[:, :, :, 1] + j, min=0, max=c - 1)

                # get the feat
                feat_grid_curr = feat_grid[:, :, grid_curr.long()].squeeze(2)

                lbs_curr = lbs[:, :, grid_curr.long()].squeeze(2).reshape(N, ncls, -1)

                # voting weight
                dist = torch.exp(- (wei * ((feat_grid_curr - feat).pow(2)).mean(1))).reshape(1, -1)

                lbs_voting = lbs_voting + lbs_curr * dist.unsqueeze(1).repeat(1, ncls, 1)

                dist_sum = dist_sum + dist

                id = id + 1

        # step 4: compute the interpolated label for grid_inte
        lbs_inte = lbs_voting / torch.clamp(dist_sum.unsqueeze(1).repeat(1, ncls, 1), min=1e-15)

        return lbs_inte.reshape(N, ncls, r_in, c_in)

# compLoss
class loss_kernelMatching_batch(nn.Module):
    def __init__(self, target):
        super(loss_kernelMatching_batch, self).__init__()
        self.target = target


    def forward(self, pred):
        a = time.time()

        bs = pred.size(0)
        losses = torch.zeros(bs, 1).cuda()

        target = self.target.float()

        for l in range(bs):
            no_pr = torch.norm(pred[l, :, :], 2)
            no_gt = torch.norm(target[l, :, :], 2)

            losses[l] = - ( pred[l, :, :] * target[l, :, :]).sum() / (no_gt * no_pr)
        #'''

        #b = time.time()

        #print(b-a)
        return (losses.sum() / bs)

# compLoss
class loss_kernelMatching_mask_batch(nn.Module):
    def __init__(self, target, mask):
        super(loss_kernelMatching_mask_batch, self).__init__()
        self.target = target
        self.mask = mask

    def forward(self, pred):
        a = time.time()

        bs = pred.size(0)
        losses = torch.zeros(bs, 1).cuda()

        target = self.target.float()
        mask = self.mask.float()

        for l in range(bs):
            idSet = mask[l, :, :].view(-1).nonzero()

            if len(idSet) > 0:
                idSet = idSet.squeeze()
                no_pr = torch.norm(pred[l, idSet, :], 2)
                no_gt = torch.norm(target[l, idSet, :], 2)

                losses[l] = - (pred[l, idSet, :] * target[l, idSet, :]).sum() / (no_gt * no_pr)
            else:
                losses[l] = 0

        if losses.sum() == 0:
            losses

        return (losses.sum() / bs)

# compLoss
class loss_kernelMatching_perpixel(nn.Module):
    def __init__(self, target):
        super(loss_kernelMatching_perpixel, self).__init__()
        self.target = target

    def forward(self, pred):
        #a = time.time()

        bs = pred.size(0)
        losses = torch.zeros(bs, 1).cuda()

        target = self.target.float()

        for l in range(bs):
            no_pr = 1 / ((pred[l, :, :].pow(2).sum(1)).pow(1/2))
            no_gt = 1 / ((target[l, :, :].pow(2).sum(1)).pow(1/2))
            losses[l]  = -((pred[l, :, :].mm(no_pr.diag())) * (target[l, :, :].mm(no_gt.diag()))).sum()

        return (losses.sum() / bs / 1681)

# compute kernel matching loss
class loss_kernelMatching_perpixel_ms(nn.Module):
    def __init__(self, target):
        super(loss_kernelMatching_perpixel_ms, self).__init__()
        self.target = target

    def forward(self, pred, list_ts):
        bs = pred.size(0)
        np = pred.size(1)
        nt = list_ts.size(0)
        losses = torch.zeros(bs, nt, np).cuda()
        target = self.target.float()

        for p in range(nt):
            for l in range(bs):
                pred_W = pred[l, :, :].pow(list_ts[p])
                targ_W = target[l, :, :]
                no_pr = 1 / ((pred_W.pow(2).sum(1)).pow(1/2))
                no_gt = 1 / ((targ_W.pow(2).sum(1)).pow(1/2))
                losses[l, p, :]  = -((pred_W.mm(no_pr.diag())) * (targ_W.mm(no_gt.diag()))).sum(1)

        loss = losses.min(1)[0]

        return (loss.sum() / bs / 1681)