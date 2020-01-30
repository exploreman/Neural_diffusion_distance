import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import SpixelAggr_avr_cuda
import numpy as np


# Feat2Dist (batch)
class Feat2Dist_function_batch(Function):
    @staticmethod
    def forward(ctx, input):
        bs, d, r, c = input.size()
        imgx = input.reshape(bs,d,r*c).permute(0,2,1).double()

        xnorm = imgx.pow(2).sum(2).reshape(bs, r*c, 1)
        vone = torch.ones(bs, r*c, 1, dtype = torch.float64).cuda()
        y = (vone.bmm(xnorm.permute(0,2,1)) + xnorm.bmm(vone.permute(0,2,1))) - 2 * imgx.bmm(imgx.permute(0,2,1))

        ctx.save_for_backward(input, vone)

        return y

    @staticmethod
    def backward(ctx, dzdy):
        input, vone = ctx.saved_variables

        bs, d, r, c = input.size()
        imgx = input.reshape(bs, d, r * c).permute(0, 2, 1).double()

        tmp = (dzdy.bmm(vone) + vone.permute(0,2,1).bmm(dzdy).permute(0, 2, 1)) #.view(-1).diag()
        w = dzdy.size(1)
        tmp0 = torch.zeros(bs, w, w).cuda().double()

        for id in range(bs):
            tmp0[id, :, :] = tmp[id, :, :].view(-1).diag()

        y = 2 * (tmp0 - dzdy - dzdy.permute(0,2,1)).bmm(imgx)

        #print(dzdy[0,:,:])
        #print(y[0,:,:])

        return y.permute(0, 2, 1).reshape(bs, -1, r, c).float()


# Feat2Dist
class Feat2Dist_function(Function):
    @staticmethod
    def forward(ctx, input):
        imgx = input.squeeze().double()
        r, c = imgx.size()
        xnorm = imgx.pow(2).sum(1).reshape(r,1)
        vone = torch.ones(r, 1, dtype = torch.float64).cuda()
        y = (vone.mm(xnorm.transpose(1,0)) + xnorm.mm(vone.transpose(1,0))) - 2 * imgx.mm(imgx.transpose(0,1))

        ctx.save_for_backward(input, vone)

        #print(y)

        return y

    @staticmethod
    def backward(ctx, dzdy):
        imgx, vone = ctx.saved_variables
        imgx = imgx.double()
        y = 2 * (dzdy.mm(vone).view(-1).diag() + vone.transpose(1,0).mm(dzdy).view(-1).diag() - dzdy - dzdy.transpose(1,0)).mm(imgx)

        tmp = dzdy.mm(vone).view(-1).diag() + vone.transpose(1,0).mm(dzdy).view(-1).diag()

        return y


class SpixelsAggr_avr_dense_function(Function):
    @staticmethod
    def forward(ctx, input, segLabels):

        #### Just for debug
        '''
        import scipy.io
        mat_file = scipy.io.loadmat('data_tmp.mat')
        input0 = mat_file['input']
        input0 = torch.from_numpy(np.asarray(input0, np.float64)).cuda().permute(2,0,1)
        input = torch.zeros(1, 128, input.size(2), input.size(3)).cuda().double()
        input[0,:,:40, :60] = input0
        ####
        '''

        segLabels = segLabels.squeeze()

        #'''
        r, d = segLabels.shape
        #range1 = torch.arange(r)
        #range0 = torch.arange(d)
        #idx = torch.arange(r*d) #range1 * d + range0

        labels_idx = segLabels.reshape(-1).double() #[idx]
        nSegs = torch.from_numpy(np.bincount(labels_idx)[1:].astype(np.float64)).cuda()  # , weights = a



        feat = input.squeeze().double() #input.size(2)-1-1

        ctx.save_for_backward(input, labels_idx, nSegs)
        output = torch.zeros(input.size(1), nSegs.size(0),  dtype=torch.float64).cuda()

        #print(output.size(), feat.size(), labels_idx.size(), nSegs.size())

        output = SpixelAggr_avr_cuda.forward(output.contiguous(), feat.contiguous(),
                                             labels_idx.contiguous(),
                                             nSegs.contiguous())


        return output[0].permute(1,0)

    @staticmethod
    def backward(ctx, grad_out):
        input, labels_idx, nSegs = ctx.saved_variables
        input = input.squeeze() #.permute((1, 2, 0))
        d, r, c = input.size()

        grad_in = torch.zeros(d, r-1, c-1).double().cuda() #-1
        grad = torch.zeros(1, d, r, c).double().cuda()
        grad_in = SpixelAggr_avr_cuda.backward(grad_in.contiguous(), grad_out.permute(1,0).contiguous(), labels_idx.contiguous(), nSegs.contiguous())

        #grad[0, :, :r-1, :c-1 ] = grad_in[0] #-1

        #print(grad_out[37, :40])
        #grad_in_tmp = grad_in[0].squeeze().reshape(2048, -1)
        #print(grad_in_tmp[:40, 1842])

        #print(grad[0, :20, 0, 0])

        return  grad_in.float(), None #,



# super-pixel aggragation
class SpixelsAggr_avr_function(Function):
    @staticmethod
    def forward(ctx, input, segLabels, coor_idx):

        #### Just for debug
        '''
        import scipy.io
        mat_file = scipy.io.loadmat('data_tmp.mat')
        input0 = mat_file['input']
        input0 = torch.from_numpy(np.asarray(input0, np.float64)).cuda().permute(2,0,1)
        input = torch.zeros(1, 128, input.size(2), input.size(3)).cuda().double()
        input[0,:,:40, :60] = input0
        ####
        '''

        segLabels = segLabels.squeeze()
        #'''
        r, d = segLabels.shape
        # idx = (coor_idx[0] - 1) * d + coor_idx[1] - 1
        six = coor_idx[0].cpu().unique().__len__()
        siy = coor_idx[1].cpu().unique().__len__()

        idx = ((coor_idx[1] - 1) * d + coor_idx[0] - 1).long()

        labels_idx = segLabels.reshape(-1)[idx].double()
        nSegs = torch.from_numpy(np.bincount(labels_idx)[1:].astype(np.float64)).cuda()  # , weights = a

        feat = input[:,:,0:siy, 0:six ].squeeze().double() #input.size(2)-1-1

        ctx.save_for_backward(input, labels_idx, nSegs, torch.tensor([siy, six]))
        output = torch.zeros(input.size(1), nSegs.size(0),  dtype=torch.float64).cuda()

        #print(output.size(), feat.size(), labels_idx.size(), nSegs.size())

        output = SpixelAggr_avr_cuda.forward(output.contiguous(), feat.contiguous(),
                                             labels_idx.contiguous(),
                                             nSegs.contiguous())


        return output[0].permute(1,0)

    @staticmethod
    def backward(ctx, grad_out):
        input, labels_idx, nSegs, si = ctx.saved_variables
        input = input.squeeze() #.permute((1, 2, 0))
        d, r, c = input.size()

        grad_in = torch.zeros(d, si[0], si[1]).double().cuda() #-1
        grad = torch.zeros(1, d, r, c).double().cuda()
        grad_in = SpixelAggr_avr_cuda.backward(grad_in.contiguous(), grad_out.permute(1,0).contiguous(), labels_idx.contiguous(), nSegs.contiguous())

        #print(grad_in[0].size())
        #print(grad.size())
        #print(si)

        grad[0, :, :si[0], :si[1] ] = grad_in[0] #-1

        #print('debug')

        #print(grad_out[37, :40])
        #grad_in_tmp = grad_in[0].squeeze().reshape(2048, -1)
        #print(grad_in_tmp[:40, 1842])

        #print(grad[0, :20, 0, 0])
        #print('for deblug:')
        #'''
        #print(torch.isnan(grad.view(-1)).sum())
        #print(grad_in[0].size())
        if (torch.isnan(grad_in[0].view(-1)).sum() > 0):
            print('nan gradient')
            print(input.size())
            print(grad_in[0].view(-1).size())
            print(torch.isnan(grad_in[0].view(-1)).nonzero())
            print(torch.isnan(grad_out.view(-1)).nonzero())
            print([labels_idx.size()])
            print(torch.nonzero(nSegs == 0))
            print(grad_out.size())
            print(grad_in[0].size())
        #'''

        return  grad.float(), None, None #,


# dist2SimiMatrix_scale (batch)
class dist2SimiMatrix_function_batch(Function):
    @staticmethod
    def forward(ctx, input, weights):
        bs,r,d = input.size()
        scale = weights
        scale_exp = torch.exp(scale)
        x = F.relu(input)
        y = torch.exp(-scale_exp * x)

        if (torch.isnan(x).sum() > 0):
            x = x

        ctx.save_for_backward(input, weights)

        return y

    @staticmethod
    def backward(ctx, dzdy):
        input, weights = ctx.saved_variables
        bs, r, d = input.size()
        scale =  weights
        scale_exp = torch.exp(scale)

        x = input
        y = dzdy * torch.exp(-scale_exp * x) * (-scale_exp)
        dw = (dzdy * (-x) * torch.exp(-scale_exp * x) * scale_exp).sum().reshape(1)

        if torch.isnan(y).sum() > 0:
            print('nan: dist2simi')
            print(torch.isnan(x).nonzero())
            print(scale_exp)

        #print(y[0,:,:])

        return y, dw


# dist2SimiMatrix_scale
class dist2SimiMatrix_function(Function):
    @staticmethod
    def forward(ctx, input, weights):
        r,d = input.size()
        scale = weights
        scale_exp = torch.exp(scale)
        x = F.relu(input)
        y = torch.exp(-scale_exp * x)

        if (torch.isnan(x).sum() > 0):
            x = x

        ctx.save_for_backward(input, weights)

        return y

    @staticmethod
    def backward(ctx, dzdy):
        input, weights = ctx.saved_variables
        r, d = input.size()
        scale =  weights
        scale_exp = torch.exp(scale)

        x = input
        y = dzdy * torch.exp(-scale_exp * x) * (-scale_exp)
        dw = (dzdy * (-x) * torch.exp(-scale_exp * x) * scale_exp).sum().reshape(1)

        if torch.isnan(y).sum() > 0:
            print('nan: dist2simi')
            print(torch.isnan(x).nonzero())
            print(scale_exp)

        #print(y[100, :10])

        return y, dw


# dist2SimiMatrix_scale
class dist2SimiMatrix_scale_function(Function):
    @staticmethod
    def forward(ctx, input, weights):
        ctx.save_for_backward(input, weights)
        r,d = input.size()
        scale = weights
        x = F.relu(input)
        ONE = torch.ones(r,1).cuda()
        xv = torch.sqrt(1 / (x * ONE)).diag()
        vx = torch.sqrt(1 / (ONE.transpose(1, 0) * x)).diag()
        y = torch.exp(-scale * xv.mm(x).mm(vx) * r)
        return y

    @staticmethod
    def backward(ctx, dzdy):
        input, weights = ctx.saved_variables
        r, d = input.size()
        scale = weights
        x = F.relu(input)
        ONE = torch.ones(r, 1)
        xv = torch.sqrt(1 / (x * ONE)).diag()
        vx = torch.sqrt(1 / (ONE.transpose(1, 0) * x)).diag()
        dzds = -scale * dzdy * r * dzdy
        y = -0.5 * (dzds * (xv.pow(3).mm(x).mm(vx)).mm(ONE).mm(ONE.transpose(0,1))) - 0.5 * ONE * ONE.transpose(1, 0).mm(dzds * (xv.mm(x).mm(vx.pow(3)))) + xv.mm(dzds.mm(vx))
        dw = (-dzdy * (xv.mm(x).mm(vx)) * r).sum()
        return y, dw

# neighMasking function (batch)
class neighMasking_function_batch(Function):
    @staticmethod
    def forward(ctx, input, neigW):
        bs = input.size(0)

        y = input * neigW.repeat(bs, 1, 1)
        ctx.save_for_backward(input, neigW)
        return y

    @staticmethod
    def backward(ctx, dzdy):
        input, neigMask = ctx.saved_variables
        bs = input.size(0)
        y = dzdy * (neigMask.repeat(bs, 1, 1))
        #import matplotlib.pyplot as plt
        #plt.imshow(y)
        #plt.show()
        return y, None

# neighMasking
class neighMasking_function(Function):
    @staticmethod
    def forward(ctx, input, neigW):
        neigW = neigW.squeeze()
        neigMask = neigW.mm(neigW).mm(neigW).mm(neigW).mm(neigW).mm(neigW)
        y = input * (((neigMask + neigMask.transpose(1,0)) > 0).double())
        ctx.save_for_backward(input, neigMask)
        return y

    @staticmethod
    def backward(ctx, dzdy):
        input, neigMask = ctx.saved_variables
        y = dzdy * ((neigMask + neigMask.transpose(1,0)) > 0).double()

        #import matplotlib.pyplot as plt
        #plt.imshow(y)
        #plt.show()
        return y, None


#compNormSimiMatrix (batch)
class compNormSimiMatrix_function_batch(Function):
    @staticmethod
    def forward(ctx, input):
        bs,r,d = input.size()
        simi = input.squeeze()
        EPS = torch.full((bs, r,1), 1e-5, dtype=torch.float64).squeeze().cuda()
        D = 1 / torch.max(simi.sum(2), EPS)

        Ddiag = torch.zeros(bs, r, r).cuda().double()
        Ddiag_sq = torch.zeros(bs, r, r).cuda().double()

        for l in range(bs):
            Ddiag[l, :, :] = D[l,:].diag()
            Ddiag_sq[l, :, :] = D[l, :].pow(2).diag()

        y = Ddiag.bmm(simi)

        ctx.save_for_backward(input, EPS, D, Ddiag, Ddiag_sq)
        return y

    @staticmethod
    def backward(ctx, dzdy):
        input, EPS, D, Ddiag, Ddiag_sq = ctx.saved_variables
        bs, r, d = input.size()

        #dzdy = dzdy * 1e+10
        #print(input.size())

        ONE = torch.ones(bs, r, 1, dtype=torch.float64).cuda()
        simi = input.squeeze()

        tmp = Ddiag_sq.bmm(simi).bmm(dzdy.permute(0, 2, 1))
        tmpDiag = torch.zeros(bs, r, 1).cuda().double()
        for l in range(bs):
            tmpDiag[l, :, 0] = tmp[l, :, :].diag()

        y = Ddiag.bmm(dzdy) - tmpDiag.bmm(ONE.permute(0,2,1))


        #print(dzdy[0,:,:])
        #print(y[0, :, :])
        return y


#compNormSimiMatrix
class compNormSimiMatrix_function(Function):
    @staticmethod
    def forward(ctx, input):
        r,d = input.size()
        #ONE = torch.ones(r, 1).cuda()
        simi = input.squeeze()
        EPS = torch.full((r,1), 1e-5, dtype=torch.float64).squeeze().cuda()
        D = 1 / torch.max(simi.sum(1), EPS)
        y = D.diag().mm(simi)
        ctx.save_for_backward(input, EPS, D)
        return y

    @staticmethod
    def backward(ctx, dzdy):
        input, EPS, D = ctx.saved_variables
        r, d = input.size()
        ONE = torch.ones(r, 1, dtype=torch.float64).cuda()
        simi = input.squeeze()

        #print(D.pow(2).diag().mm(simi).mm(dzdy.transpose(1,0)).diag().size())
        y = D.diag().mm(dzdy) - (D.pow(2).diag().mm(simi).mm(dzdy.transpose(1,0))).diag().reshape(-1, 1).mm(ONE.transpose(1,0))

        print(dzdy[:,:])
        print(y[:, :])


        return y

# comp approximate eigendecomposition

class compQRDecomp_function(Function):

    #def bsym(self, input):
        #return input.tril() - (input.tranpose(1,0)).tril()

    @staticmethod
    def forward(ctx, input):
        [bs, r, d] = input.size()

        Q = torch.zeros(bs, r, d, dtype = torch.float64).double().cuda()
        R = torch.zeros(bs, d, d, dtype = torch.float64).double().cuda()
        for id in range(bs):
            Q[id, :, :], R[id, :, :] = torch.qr(input[id, :, :].squeeze())

        ctx.save_for_backward(input, Q, R)

        return Q, R

    @staticmethod
    def backward(ctx, dQs, dRs):
        input, Qs, Rs = ctx.saved_variables
        [bs, r, d] = input.size()
        X = torch.zeros_like(input).double().cuda()

        #print(dQs.size(), dRs.size())
        for id in range(bs):
            Q = Qs[id, :, :].squeeze()
            R = Rs[id, :, :].squeeze()
            dR = dRs[id, :, :].squeeze()
            dQ = dQs[id, :, :].squeeze()
            S = torch.eye(r, r).double().cuda() - Q.mm(Q.transpose(1,0))
            RInv = torch.inverse(R).transpose(1,0)
            tmpQ = Q.transpose(1,0).mm(dQ)
            tmpR = dR.mm(R.transpose(1,0))
            X[id, :, :] = (S.transpose(1,0).mm(dQ) + Q.mm(tmpQ.tril() - (tmpQ.transpose(1,0)).tril())).mm(RInv)  + Q.mm(dR - (tmpR.tril() - (tmpR.transpose(1,0)).tril()).mm(RInv))

        return X


class compQRDecomp_function_f(Function):

    #def bsym(self, input):
        #return input.tril() - (input.tranpose(1,0)).tril()

    @staticmethod
    def forward(ctx, input):
        [bs, r, d] = input.size()

        Q = torch.zeros(bs, r, d).cuda()
        R = torch.zeros(bs, d, d).cuda()
        #EPS = 1e-2 * torch.ones(d).cuda()

        for id in range(bs):
            Q[id, :, :],R[id, :, :]= torch.qr(input[id, :, :].squeeze())
            #R_diag = R_tmp.diag()
            #R[id, :, :].view(-1)[::d+1] = (R_diag == 0).float() * 1e-2 + (R_diag != 0).float() * R_diag

            #try:
            #    torch.inverse(R[id, :, :])
            #except:
            #    R


        ctx.save_for_backward(input, Q, R)

        return Q, R

    @staticmethod
    def backward(ctx, dQs, dRs):
        input, Qs, Rs = ctx.saved_variables
        [bs, r, d] = input.size()
        X = torch.zeros_like(input).cuda()

        #print(dQs.size(), dRs.size())
        for id in range(bs):
            Q = Qs[id, :, :].squeeze()
            R = Rs[id, :, :].squeeze()
            dR = dRs[id, :, :].squeeze()
            dQ = dQs[id, :, :].squeeze()
            S = torch.eye(r, r).cuda() - Q.mm(Q.transpose(1,0))
            RInv = torch.inverse(R).transpose(1,0)

            tmpQ = Q.transpose(1,0).mm(dQ)
            tmpR = dR.mm(R.transpose(1,0))
            X[id, :, :] = (S.transpose(1,0).mm(dQ) + Q.mm(tmpQ.tril() - (tmpQ.transpose(1,0)).tril())).mm(RInv)  + Q.mm(dR - (tmpR.tril() - (tmpR.transpose(1,0)).tril()).mm(RInv))

        return X

# compEigDecomp
class compEigDecomp_function(Function):

    @staticmethod
    def forward(ctx, input):
        x = input.squeeze()
        r, d = x.size()

        G, V = torch.eig(x, eigenvectors=True)

        G = G[:, 0].view(-1)
        tmp, order = torch.sort(G, 0, descending=True)
        eigG = G[order].diag()
        eigV = V[:, order]
        ctx.save_for_backward(input, eigG.diag(), eigV)
        return eigG.diag(), eigV

    @staticmethod
    def backward(ctx, dzdG, dzdV):
        input, eigG, eigV = ctx.saved_variables

        dzdG = dzdG.diag()
        x = input.squeeze()
        r, d = x.size()
        n_neig = dzdG.size(1)
        vone = torch.ones(r, 1, dtype=torch.float64).cuda()

        if n_neig < r:
            dzdV = torch.cat((dzdV, torch.zeros(r, r - n_neig, dtype=torch.float64).cuda()), 1)
            dzdG = torch.cat((dzdG.diag().reshape(1,-1), torch.zeros(1, r-n_neig, dtype=torch.float64).cuda()), 1).view(-1).diag()

        eigG_d2 = eigG.reshape(-1,1)

        diff = eigG_d2.mm(vone.permute(1,0))-(eigG_d2.mm(vone.permute(1,0))).permute(1,0)

        K = 1 / diff
        K[torch.eye(K.size(0)) > 0] = 0

        K[torch.isnan(K)] = 0
        K[K == float("Inf")] = 0

        tmp = (eigV.permute(1, 0).mm(dzdV))
        y = eigV.mm(dzdG).mm(eigV.permute(1,0)) + eigV.mm(K.permute(1,0) * tmp).mm(eigV.permute(1,0))

        if torch.isnan(y).sum() > 0:
            print('nan: eig')

        #import matplotlib.pyplot as plt
        #plt.imshow(y)
        #plt.show()
        #print(y[:, :])

        return y


# compDiffDist (batch)
class compDiffDist_function(Function):
    @staticmethod
    def forward(ctx, G, V, t):
        #G, V = input
        #Gs = G.diag()
        G = G.diag()
        r, d = V.size()
        n_eigs = min(100,d)
        st = 1
        G = torch.max(G, torch.zeros_like(G))
        E = torch.ones(r, n_eigs - st + 1, dtype = torch.float64).cuda()
        Phi = V[:, st-1: n_eigs]
        Phi2 = Phi.pow(2)
        eVs = ((G[st-1: n_eigs, st-1: n_eigs]).diag().pow(2 * t)).diag()
        y = E.mm(eVs).mm(Phi2.transpose(1,0)) + Phi2.mm(eVs).mm(E.transpose(1,0)) - 2 * Phi.mm(eVs).mm(Phi.transpose(1,0))
        ctx.save_for_backward(G, V, t)

        return y

    @staticmethod
    def backward(ctx, dzdy):
        G, V, t = ctx.saved_variables
        r,d = V.size()
        n_eigs = min(100,d)

        vone = torch.ones(r, 1, dtype=torch.float64).cuda()
        st = 1
        dzdt = 0
        Phi = V[:, st-1: n_eigs]
        Gs = G

        y = 2 * ((dzdy.transpose(1,0).mm(vone).squeeze().diag()).mm(Phi) + (dzdy.mm(vone)).squeeze().diag().mm(Phi) - dzdy.transpose(1,0).mm(Phi) - dzdy.mm(Phi)).mm(G[st-1:n_eigs, st-1: n_eigs].pow(2 * t))
        Ks = torch.zeros(r * r, n_eigs - st + 1, dtype=torch.float64).cuda()
        for p in range(n_eigs - st + 1):
            data = Phi[:,p].reshape(-1,1)
            tmp = vone.mm(data.pow(2).transpose(1,0)) +  data.pow(2).mm(vone.transpose(1,0)) - 2 * data.mm(data.transpose(1,0))
            Ks[:,p] = (2 * t[0] * Gs[p, p].pow(2 * t[0] - 1) * tmp).transpose(1,0).contiguous().view(-1)
            dzdt = dzdt + 2 * torch.log(torch.max(Gs[p, p], torch.tensor([1e-10], dtype=torch.float64).cuda())) * (Gs[p, p].pow(2 * t[0])) * tmp
        vdzdy = dzdy.reshape(1, dzdy.size(1) * dzdy.size(0))
        z = vdzdy.mm(Ks)
        dzdt = vdzdy.mm(dzdt.reshape(-1,1))
        z = z.squeeze().diag()

        #print(dzdt)

        if torch.isnan(z).sum() > 0 or torch.isnan(y).sum() > 0:
            print('nan:diff')

        #print(y[100,:20])
        #import matplotlib.pyplot as plt
        #plt.imshow(y)
        #plt.show()

        #print(dzdy[:, :])
        return z.diag(), y, dzdt.view(-1)


# compDiffDist
class compDiffDist_function_batch(Function):
    @staticmethod
    def forward(ctx, G, V, t):
        #G = G.diag()
        bs, r, d = V.size()
        G = torch.max(G, torch.zeros_like(G))
        #eVs = G
        E = torch.ones(bs, r, d, dtype = torch.float64).cuda()
        eVs = torch.ones(bs, d, d, dtype = torch.float64).cuda()

        Phi = V
        Phi2 = V.pow(2)

        for l in range(bs):
            eVs[l, :, :] = G[l, :].pow(2 * t).diag()

        y = E.bmm(eVs).bmm(Phi2.permute(0, 2, 1)) + Phi2.bmm(eVs).bmm(E.permute(0,2,1)) - 2 * Phi.bmm(eVs).bmm(Phi.permute(0, 2,1))
        ctx.save_for_backward(G, V, t)

        return y

    @staticmethod
    def backward(ctx, dzdy):
        G, V, t = ctx.saved_variables
        bs,r,d = V.size()

        vone = torch.ones(bs, r, 1, dtype=torch.float64).cuda()
        tmp0 = torch.ones(bs, r, r, dtype=torch.float64).cuda()

        dzdt = 0
        Phi = V
        Gs = G

        tmp = dzdy.permute(0, 2, 1).bmm(vone) + (dzdy.bmm(vone)) #.squeeze().diag())
        Gdiag = torch.ones(bs, d, d, dtype=torch.float64).cuda()

        for l in range(bs):
            tmp0[l, :, :] = tmp[l,:, :].squeeze().diag()
            Gdiag[l, :, :] = G[l,:].diag()

        y = 2 * (tmp0 - dzdy.permute(0,2,1) - dzdy).bmm(Phi).bmm(Gdiag.pow(2 * t))

        vdzdy = dzdy.reshape(bs, 1, dzdy.size(2) * dzdy.size(1))
        z = torch.zeros(bs, d, dtype=torch.float64).cuda()
        for p in range(d):
            data = Phi[:,:,p].reshape(bs,-1, 1)
            tmp = vone.bmm(data.pow(2).permute(0,2,1)) +  data.pow(2).bmm(vone.permute(0,2,1)) - 2 * data.bmm(data.permute(0,2,1))

            Gval = Gs[:,p].reshape(bs, 1, 1)
            z[:, p] = vdzdy.bmm((2 * t[0] * Gval.pow(2 * t[0] - 1)).mul(tmp).permute(0, 2, 1).reshape(bs, r * r, 1)).squeeze()
            dzdt = dzdt + 2 * torch.log(torch.max(Gval, torch.tensor([1e-10], dtype=torch.float64).cuda())) * (Gval.pow(2 * t[0])).mul(tmp)

        dzdt = vdzdy.bmm(dzdt.reshape(bs,-1,1)).sum()

        if torch.isnan(z).sum() > 0 or torch.isnan(y).sum() > 0:
            print('nan:diff')

        return z.squeeze(), y, dzdt.view(-1)

# to be updated for the formulation
'''
class kernelMatching_function_batch(Function):

    @staticmethod
    def forward(ctx, predict, target):
        #target = 1 - target.squeeze().float()
        target = target.double().squeeze()
        #assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        #assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        #n = predict.size(0)
        # target_mask = (target >= 0) * (target != self.ignore_label)
        # target = target[target_mask]
        # if not target.data.dim():
        #    return Variable(torch.zeros(1))
        # predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        # predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        # loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        loss = 0
        bs = predict.size(0)
        no_pr = torch.zeros(bs).cuda().double()
        no_gt = torch.zeros(bs).cuda().double()

        for l in range(bs):
            no_pr[l] = torch.norm(predict, 2)
            no_gt[l] = torch.norm(target, 2)

            loss = loss - (predict[l, :, :] * target[l, :, :]).sum() / (no_gt[l] * no_pr[l])

        ctx.save_for_backward(predict, target, no_pr, no_gt)

        return loss

    @staticmethod
    def backward(ctx, dzdl):
        # target_mask = (target >= 0) * (target != self.ignore_label)
        # target = target[target_mask]
        # if not target.data.dim():
        #    return Variable(torch.zeros(1))
        # predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        # predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        # loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)

        #gt = (target / no_gt).reshape(-1, 1)
        #x = predict.reshape(-1, 1)
        #grad = dzdl * (-1 / no_pr * (gt - x * (x.transpose(1, 0).mm(gt)) / no_pr.pow(2))).reshape(target.size())

        # grad = []
        predict, target, no_pr, no_gt = ctx.saved_variables
        bs = target.size(0)
        x = predict.reshape(bs, -1, 1)
        gt = torch.zeros_like(x).cuda()

        #print(target.size(), gt.size(), no_gt.size(), (target[0, :, :] / no_gt[0]).size())
        for l in range(bs):
            gt[l, :] = (target[l, :, :] / no_gt[l]).reshape(-1,1)

        no_pr = no_pr.reshape(bs, 1, 1)

        #print([(-1 / no_pr).size(), x.size(), gt.size()]) # (x.permute(0, 2, 1).bmm(gt)).size(0))
        grad = dzdl * ((-1 / no_pr).mul(gt - x * (x.permute(0, 2, 1).bmm(gt)).mul(1 / no_pr.pow(2)))).reshape(target.size())

        #print(grad)
        #print(gt.type())
        #print(grad[440,:10])
        #import matplotlib.pyplot as plt
        #plt.imshow(grad)
        #plt.show()


        if torch.isnan(grad).sum() > 0:
            print('nan')


        return grad, None
'''

# to be updated for the formulation
class kernelMatching_function(Function):

    @staticmethod
    def forward(ctx, predict, target):
        #target = 1 - target.squeeze().float()
        target = target.double().squeeze()
        #assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        #assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        #n = predict.size(0)
        # target_mask = (target >= 0) * (target != self.ignore_label)
        # target = target[target_mask]
        # if not target.data.dim():
        #    return Variable(torch.zeros(1))
        # predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        # predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        # loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        no_pr = torch.norm(predict, 2)
        no_gt = torch.norm(target, 2)
        loss = - (predict * target).sum() / (no_gt * no_pr)
        ctx.save_for_backward(predict, target, no_pr, no_gt)

        return loss

    @staticmethod
    def backward(ctx, dzdl):
        # target_mask = (target >= 0) * (target != self.ignore_label)
        # target = target[target_mask]
        # if not target.data.dim():
        #    return Variable(torch.zeros(1))
        # predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        # predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        # loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)

        # grad = []
        predict, target, no_pr, no_gt = ctx.saved_variables
        gt = (target / no_gt).reshape(-1, 1)
        x = predict.reshape(-1, 1)
        grad = dzdl * (-1 / no_pr * (gt - x * (x.transpose(1, 0).mm(gt)) / no_pr.pow(2))).reshape(target.size())

        #print(no_pr)
        #print(gt.type())
        #print(grad[440,:10])
        #import matplotlib.pyplot as plt
        #plt.imshow(grad)
        #plt.show()


        if torch.isnan(grad).sum() > 0:
            print('nan')

        return grad, None


# codes for gradient checking
if __name__ == '__main__':
    # check gradient
    from torch.autograd import gradcheck
    from torch.autograd import Variable

    # check loss layer
    #'''
    input = (
        Variable(100 * torch.randn(2, 20, 20).double().cuda(), requires_grad=True),
        Variable(100 * torch.randn(2, 20, 20).double().cuda(), requires_grad=False),
    )
    test = gradcheck(kernelMatching_function_batch.apply, input, eps=1e-6, atol=1e-4)
    print(test)
    #'''

    # check for QR decomposition
    tmp = torch.rand(2, 3, 3).double()
    input = (
        Variable(tmp.bmm(tmp.permute(0, 2, 1)).double().cuda(), requires_grad=True),
    )
    test = gradcheck(compQRDecomp_function.apply, input, eps=1e-4, atol=1e-4)
    print(test)


    # check gradient for super-pixel_pooling layer
    # (ctx, input, segLabels, coor_idx):
    '''
    lab = torch.ones(2, 2).double().cuda()
    lab[0:1, 0:] = 2
    fea = torch.rand(1, 5, 2, 2).double().cuda()
    r = range(1, 3, 1)
    c = range(1, 3, 1)
    grid = np.meshgrid(c, r)
    grid = np.stack(grid)
    coor_idx = torch.from_numpy(grid.reshape(2, -1)).cuda()
    input = (
        Variable(fea.float().cuda(), requires_grad=True),
        Variable(lab.double().cuda(), requires_grad=False),
        Variable(coor_idx.cuda(), requires_grad=False),
    )
    test = gradcheck(SpixelsAggr_avr_function.apply, input, eps=3e-5, atol=1e-3)
    print(test)
    #out = SpixelAggr_avr_cuda.forward(out.contiguous(), fea.contiguous(), lab.contiguous(), nSeg.contiguous())
    #print('______')
    #print(fea)
    #print('______')
    #print(out)
    #print('______')
    #print(lab)
    '''

    # check diff_distance layer
    ''' FINE
    tmp = torch.rand(2,10,10).double()
    input = (
        Variable(torch.rand(2,10).double().cuda(), requires_grad=True),
        Variable(tmp.bmm(tmp).double().cuda(),requires_grad=True),
        Variable(torch.tensor([5]).double().cuda(), requires_grad=True),
    )
    test = gradcheck(compDiffDist_function_batch.apply, input, eps=1e-8, atol=1e-4)
    print(test)
    '''

    # check compEigDecomp_function
    ''' ???
    tmp = torch.rand(2,3,3).double()
    input = (
         Variable(tmp.bmm(tmp.permute(0, 2,1)).double().cuda(),requires_grad=True),
    )
    test = gradcheck(compEigDecomp_function_batch.apply, input, eps=1e-4, atol=1e-4)
    print(test)
    '''

    #check compNormSimiMatrix_function layer
    ''' FINE!
    tmp = torch.rand(2, 5, 5).double()
    input = (
        Variable(tmp.bmm(tmp).double().cuda(), requires_grad=True),
    )
    test = gradcheck(compNormSimiMatrix_function_batch.apply, input, eps=1e-6, atol=1e-4)
    print(test)
    '''

    # check masking layer
    ''' FINE
    tmp = torch.rand(5, 5).double()
    input = (
        Variable(tmp.mm(tmp).double(), requires_grad=True),
        Variable(torch.rand(5,5).double() > 0, requires_grad=True),
    )
    test = gradcheck(neighMasking_function.apply, input, eps=1e-6, atol=1e-4)
    print(test)
    '''

    # check dist2SimiMatrix_function layer
    ''' FINE
    tmp = torch.rand(1, 5, 5).double()
    input = (
        Variable(tmp.bmm(tmp).double().cuda(), requires_grad=True),
        Variable(torch.rand(1).double().cuda(), requires_grad=True),
    )
    test = gradcheck(dist2SimiMatrix_function_batch.apply, input, eps=1e-6, atol=1e-4)
    print(test)
    '''


    # check Feat2Dist_function layer
    ''' FINE
    tmp = torch.rand(1, 2, 3, 3).double().cuda()

    input = (
        Variable(tmp.squeeze().reshape(2,9).transpose(1,0).double().cuda(), requires_grad=True),
    )

    test = gradcheck(Feat2Dist_function.apply, input, eps=1e-6, atol=1e-4)
    print(test)
    '''


    # check Feat2Dist_function_batch layer
    ''' FINE
    #tmp = torch.rand(1, 2, 5, 5).double().cuda()
    #input = (
    #    Variable(tmp.mm(tmp).double().cuda(), requires_grad=True),
    #)

    input = (
        Variable(tmp.double().cuda(), requires_grad=True),
    )

    test = gradcheck(Feat2Dist_function_batch.apply, input, eps=1e-6, atol=1e-4)
    print(test)
    '''


