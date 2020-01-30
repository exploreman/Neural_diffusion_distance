#####################################
# import required toolboxes
#####################################
import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from diffMap_deeplab.model import DiffNet_grid
from diffMap_deeplab.model import Upsample_fullScale

from diffMap_deeplab import BSDataSet_mask
from diffMap_deeplab.diffMap_layers_f import loss_kernelMatching_batch #, loss_kernelMatching
from diffMap_deeplab.diffMap_layers_f import loss_kernelMatching_mask_batch #, loss_kernelMatching

import timeit
start = timeit.default_timer()
import time
from weighted_cross_entropy import CrossEntropyLoss

##########################################
#  Parameter and folder setup
##########################################
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
BATCH_SIZE = 3

# The path to training dataset
DATA_DIRECTORY = '../Dataset/BSD500/ALL_new_orig/'

# The path to text file including the filename of training images
DATA_LIST_PATH =  './diffMap_deeplab/data/trainval.txt'
INPUT_SIZE = '321,321'
LEARNING_RATE = 0.8e-6
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 120000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM =  './snapshots_diffMap_grid/BSD_diffMap120000_clus_BSD500_trainval.pth'  # set to the pre-trained network:'./initModel/MS_DeepLab_resnet_pretrained_COCO_init.pth'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 4000
SNAPSHOT_DIR = './snapshots_diffMap_grid/'
WEIGHT_DECAY = 0

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()

def loss_kerkmeans_ss(pred_W, label_clusters):
    ###### input: pred_W, label_clusters. single scale (kernel kmeans)

    nsamples = label_clusters.size(0)
    nps = pred_W.size(1)
    si_grid = int(np.sqrt(nps))
    CE = CrossEntropyLoss() #torch.nn.
    total_loss = 0
    max_ncls = 100
    dists = torch.zeros(max_ncls, nps).cuda()

    for t in range(nsamples):
        ## step 1: randomly sample centers from each cluster in g.t. labeled cluster
        W = pred_W[t, :, :]

        with torch.no_grad():
            label_cluster = label_clusters[t, :, :]

            lbs = label_cluster.unique() # a list of labels
            ncls = lbs.size(0)

            scale = 2e+1
            centers = torch.zeros(ncls, 1)
            label_cluster_rs = torch.zeros_like(label_cluster).view(-1).cuda()
            id_cum = 0
            mask = torch.zeros_like(label_cluster_rs)
            for i in range(len(lbs)):
                idCluster = torch.nonzero(label_cluster.view(-1) == lbs[i])
                if idCluster.size(0) > 5:
                    centers[id_cum] = idCluster[(idCluster.size(0) * torch.rand(1, 1)).long()][0][0]
                    label_cluster_rs[idCluster] = id_cum
                    mask[idCluster] = 1
                    id_cum = id_cum + 1

            ncls = id_cum
            label_cluster = label_cluster_rs.reshape(si_grid, si_grid)
            centers = centers[:ncls]

        ## step 2: run fixed steps of kernel kmeans
        tmp = torch.exp(scale * W[centers.long(), :]).squeeze()
        assignMat = tmp / (tmp.sum(0)).repeat(ncls, 1)  # soft assignment matrix

        maxIter = 3
        for i in range(maxIter):
            Nk = assignMat.sum(1)

            # compute distance
            for k in range(ncls):
                prob = assignMat[k, :]
                dists[k, :] = W.diag() - (2 / (Nk[k])) * (W.matmul(prob.diag()).sum(1)) + Nk[k].pow(-2) * ((prob.unsqueeze(1)) * (prob.unsqueeze(1).permute(1,0)) * W).sum()

            # update assignMat
            tmp2 = torch.exp(-dists[0:ncls,:] * scale)
            assignMat = tmp2 / (tmp2.sum(0).repeat(ncls, 1))

        # step 3: caculate the loss between assignment matrix and g.t. cluster labels
        loss =  CE(assignMat.permute(1, 0), label_cluster.view(-1, 1).squeeze().long().cuda(), weights=mask.float().cuda())
        total_loss += loss.data[0]

    return total_loss

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion1 = loss_kernelMatching_batch(label).cuda() # loss_kernelMatching_batch loss_kernelMatching_perpixel
    #criterion2 = loss_kernelMatching_perpixel(label).cuda()
    
    return (criterion1(pred)) #+ criterion2(pred)) * 0.5


def loss_calc_mask(pred, label, mask):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion1 = loss_kernelMatching_mask_batch(label, mask).cuda()  # loss_kernelMatching_batch loss_kernelMatching_perpixel
    # criterion2 = loss_kernelMatching_perpixel(label).cuda()

    return (criterion1(pred))  # + criterion2(pred)) * 0.5



def loss_calc_hiReso(pred, gt_clusters, gt_lst, feat_hirc):

    nSmps = 20
    N = gt_clusters.size(0)
    r = gt_clusters.size(1)
    c = gt_clusters.size(2)

    # randomnly sampling points in the lowest reso, and fetch corresponding conf. map at lowest level, and g.t. map at highest resolution
    np_lst = pred.size(1)
    w_lst = int(np_lst**(1/2))
    smpls = (torch.rand(1, nSmps) * (np_lst-1)).round().long()
    lb_est_lst = pred[:, smpls, :].reshape(-1, smpls.size(1), w_lst, w_lst)

    # upsampling the lb map to full resolution
    Usmap_fun = Upsample_fullScale()
    lb_upsampl = Usmap_fun(feat_hirc, lb_est_lst)

    # measure the similarity by what? Concerntrate more on boundary or not
    loss = 0
    for n in range(N):
        lb_list = gt_clusters[n,:,:].view(-1).unique()
        ncls = lb_list.size(0)
        map = torch.zeros(ncls, r, c).cuda()

        for l in range(ncls):
            map[l,:,:] = gt_clusters[n, :,:] == l + 1

        lb_upsamp = gt_lst.view(-1)[smpls]
        lbs_gt = map[lb_upsamp.long(),:,:]
        loss = loss + torch.abs(lb_upsampl - lbs_gt[:,:,0::2, 0::2]).mean()  # L1-norm
        return loss



def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)
    b.append(model.layer5)

    
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer6.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i
            
            
def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10

def makeneigMask(r,d):

    neigMask = torch.eye(r * d, r * d).double()

    for i in range(r):
        for j in range(d):
            tmp = torch.zeros(r,d)

            for p in range(-1, 2):
                for q in range(-1, 2):
                    x = i + p
                    y = j + q
                    if x < 0:
                        x = 0
                    if x > r-1:
                        x = r-1
                    if y < 0:
                        y = 0
                    if y > d-1:
                        y = d-1
                    tmp[x, y] = 1
            neigMask[i * d + j] = tmp.view(-1)

    neigMask = neigMask.mm(neigMask).mm(neigMask).mm(neigMask) #.mm(neigMask) #.mm(neigMask) #g.mm(neigMask).mm(neigMask)
    neigMask = neigMask.mm(neigMask)
    neigMask = ((neigMask + neigMask.transpose(1, 0)) > 0).float() #.double()

    return neigMask


def main():
    """Create the model and start the training."""
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    # Create network.
    model = DiffNet_grid()
    saved_state_dict = torch.load(args.restore_from)

    if 0:
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not args.num_classes == 21 or not i_parts[1]=='layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)
        st = 0
    else:
        model.load_state_dict(saved_state_dict)
        st = 120000

    model.train()
    model.cuda()
    
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(BSDataSet_mask(args.data_dir, args.data_list, max_iters=args.num_steps*args.batch_size, mirror=True,
                                            mean=IMG_MEAN, scale=True), batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    optimizer = optim.Adam([{'params': get_1x_lr_params_NOscale(model), 'lr': 1 * args.learning_rate },
                {'params': get_10x_lr_params(model), 'lr': 5e-2 * args.learning_rate}],
                lr=args.learning_rate,  weight_decay=args.weight_decay)

    optimizer.zero_grad()

    interp = nn.Upsample(size=input_size, mode='bilinear')
    loss_avr = 0

    # compute the neightMask online
    r,d = input_size
    r = int(r / 8) + 1
    d = int(d / 8) + 1
    neigMask = makeneigMask(r,d)
    lamb = 1

    for i_iter, batch in enumerate(trainloader):
        iter = st + i_iter
        images, gt_Ws, mask, gt_clusters, gt_lst = batch
        images = Variable(images).cuda()
        neigMask = neigMask.cuda()
        gt_Ws = gt_Ws.cuda()

        a = time.time()

        inputs = (images, neigMask)
        pred, pred_raw, feat_hirc = model(inputs)

        # calculate the loss (kernel matching + clustering results)
        # loss_km = loss_calc(pred, gt_Ws)
        loss = loss_calc_mask(pred, gt_Ws, mask)

        loss.backward()

        if i_iter % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter)

        b = time.time()
        loss_avr = (loss_avr * (i_iter) + loss.data.cpu().numpy()) / (i_iter + 1)


        print('iter = ', iter, 'of', args.num_steps,'completed, loss = ', [loss_avr, loss.data.cpu().numpy()], 'time = ', b- a )

        if iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'BSD_diffMap'+str(args.num_steps)+'_clus_BSD500_trainval.pth')) #_trainval
            break

        if iter % args.save_pred_every == 0 and i_iter!=0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'BSD_diffMap'+str(iter)+'_clus_BSD500_trainval.pth')) #_trainval

    end = timeit.default_timer()
    print (end-start,'seconds')

if __name__ == '__main__':
    main()


'''
mat_name = ['tmp_diff.mat'][0]
scipy.io.savemat(mat_name, {'kernel': pred_W.cpu().detach().numpy()})

import matplotlib.pyplot as plt
plt.imshow(assignMat.max(0)[1].reshape(41,41))
plt.show()

import matplotlib.pyplot as plt
plt.imshow(pred[0, 21*41+21,:].reshape(41,41).cpu().detach())
plt.show()

plt.imshow(label_cluster)
plt.show()

import matplotlib.pyplot as plt

if 1:
    plt.imshow(pred[0, 21*41+18,:].reshape(41,41).pow(1).cpu().detach())
    plt.show()
    
    plt.imshow(images[0,0,:,:].cpu().detach())
    plt.show()
else:
    plt.imshow(x[0,:,0].reshape(41,41).pow(1).cpu().detach())
    
    
import matplotlib.pyplot as plt
a = 21
b = 16
if 1:
    plt.imshow(pred[0, a * 41 + b, :].reshape(41, 41).pow(1).cpu().detach())
    plt.show()
    plt.imshow(gt_Ws[0, a * 41 + b, :].reshape(41, 41).pow(1).cpu().detach())
    plt.show()
    plt.imshow(images[0, 0, :, :].cpu().detach())
    plt.show()
else:
    plt.imshow(x[1, :, 3].reshape(41, 41).pow(1).cpu().detach())
    plt.show()
    print(t[l, :, :].diag())
    
'''