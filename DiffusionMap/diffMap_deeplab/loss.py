import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


# to be updated for the formulation
class kernelMatchingLoss(nn.Module):

    def __init__(self):
        super(kernelMatchingLoss, self).__init__()

    @staticmethod
    def forward(ctx, predict, target, weight=None):
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        n = predict.size(0)
        #target_mask = (target >= 0) * (target != self.ignore_label)
        #target = target[target_mask]
        #if not target.data.dim():
        #    return Variable(torch.zeros(1))
        #predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        #predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        #loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        no_pr = torch.norm(predict, 2)
        no_gt = torch.norm(target, 2)
        loss = - (predict * target).sum() / (no_gt * no_pr)
        ctx.save_for_forward(predict, target, no_pr, no_gt)

        return loss

    @staticmethod
    def backward(ctx, dzdl):

        #target_mask = (target >= 0) * (target != self.ignore_label)
        #target = target[target_mask]
        #if not target.data.dim():
        #    return Variable(torch.zeros(1))
        #predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        #predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        #loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)

        #grad = []
        predict, target, no_pr, no_gt = ctx.saved_variables
        no_pr, no_gt = ctx.saved_variables
        gt = target / no_gt
        x = predict
        grad = dzdl * (-1 / no_pr * (gt - x.mm(x.transpose(1,0).mm(gt))) / no_pr.pow(2)).reshape(x.size())

        return grad