import argparse
from diffMap_deeplab.model import Upsample_fullScale
from diffMap_deeplab.model import Upsample_fullScale_cpu
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import skimage.morphology
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from diffMap_deeplab.model import DiffNet_grid
from deeplab.datasets import VOCDataTestSet
from diffMap_deeplab import BSDataTestSet
from diffMap_deeplab.diffMap_layers_f import loss_kernelMatching_batch #, loss_kernelMatching
from diffMap_deeplab.diffMap_layers_f import loss_kernelMatching_perpixel
import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()
import time
from weighted_cross_entropy import CrossEntropyLoss

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 1
isVOC = 0
if isVOC:
    DATA_DIRECTORY = '/home/jiansun/Dataset/VOC/VOC2012/VOCdevkit/VOC2012/' #'/home/jiansun/Dataset/BSD500/ALL_new_orig/' #
    #DATA_LIST_PATH = '/home/jiansun/Dataset/VOC_dataSet/train_aug_valid.txt' #'./diffMap_deeplab/data/list.txt'
    DATA_LIST_PATH = '/home/jiansun/Dataset/VOC/scribble_annotation/list_scribbel_tmp.txt'  # './diffMap_deeplab/data/list.txt'

    FOLDER_SAVE = '/home/jiansun/Projects/Datasets/DiffusionMap/clusterMaps_voc_test/'
else:
    DATA_DIRECTORY = '/home/jiansun/Dataset/BSD500/ALL_new_orig/' #
    DATA_LIST_PATH = './diffMap_deeplab/data/test.txt' #test.txt' #test.txt' # #'./diffMap_deeplab/data/list.txt' #
    FOLDER_SAVE = '/home/jiansun/Projects/Datasets/DiffusionMap/clusterMaps_bsd_test_full/' #_v2


IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2e-6
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 200 #310000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './snapshots_diffMap_grid/BSD_diffMap164000_clus_BSD500_trainval_full.pth' # -- full case

SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots_diffMap_grid/'
WEIGHT_DECAY = 0 #0.0005




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
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
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

def kerkmeans_ss(pred_W, ncls):
    ###### input: pred_W, label_clusters. single scale (kernel kmeans)

    nsamples = ncls
    nps = pred_W.size(1)
    si_grid = int(np.sqrt(nps))

    dists = torch.zeros(ncls, nps).cuda()

    ## generate the samples


    # run kernel kmeans
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
    criterion = loss_kernelMatching_perpixel(label).cuda() #loss_kernelMatching_batch
    
    return criterion(pred)


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

def  makeneigMask(r,d):

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

    '''
    basis = torch.zeros(r * d * r * d, 1)
    basis[1::r * d + 1] = 1
    basis[d - 1::r * d + 1] = 1
    basis[d::r * d + 1] = 1
    basis[d + 1::r * d + 1] = 1

    neigMask = neigMask + basis.reshape(r * d, r * d)
    neigMask = ((neigMask + neigMask.transpose(1, 0)) > 0).double()

    neigMask[0:r, r: r * d - 1] = 0
    neigMask[r * d - r:r * d - 1, 0:r] = 0
    '''

    neigMask = neigMask.mm(neigMask).mm(neigMask).mm(neigMask) #.mm(neigMask) #.mm(neigMask) #g.mm(neigMask).mm(neigMask)
    neigMask = neigMask.mm(neigMask)
    neigMask = ((neigMask + neigMask.transpose(1, 0)) > 0).float() #.double()

    return neigMask

def kernel_kmeans(images, Ke, si, maxCls):
    # inputs: images, pred, ncls, seglabel
    r = si[0]
    d = si[1]
    Ke = Ke.squeeze()
    N = Ke.size(0)

    converged = 0
    #Prob = torch.zeros(N, K)
    scale = 5e+1

    with torch.no_grad():
        # randomly generate cluster centers
        if 0:
            maxIter = 20
            centers = torch.zeros(maxIter, K)
            coverage = torch.zeros(maxIter, 1)
            for it in range(maxIter):
                ids = (torch.rand(K, 1) * (N - 1)).round().long()
                map = Ke[ids, :].sum(0).reshape(r, d)
                coverage[it] = torch.min(map, torch.ones_like(map)).sum()
                centers[it, :] = ids.squeeze()

            b=coverage.max(0)[1]
            center_best = centers[b, :].squeeze().long()

        else:
            # iteratively generate cluster center by farthest sampling
            #maxCls = 20
            centers = torch.zeros(maxCls, 1) #.cuda()
            coverage = torch.zeros(r,d) #.cuda()
            tones = torch.ones(r,d) #.cuda()
            maxIter = 10
            iter = 1
            centers[0] = round((r * d) / 2) + 1
            coverage = Ke[centers[0].long(), :].reshape(r,d)
            while coverage.sum() < 0.98 * r * d and iter < maxIter-1:
                resi = tones - coverage
                #[a, id] = resi.view(-1).max(0)
                id = Ke.matmul(resi.reshape(-1,1)).argmax()
                centers[iter] = id
                coverage = coverage + Ke[id, :].reshape(r,d)
                coverage = torch.min(coverage, tones)
                iter = iter + 1

            center_best = centers[:iter].squeeze().long()
            K = iter

        tmp = torch.exp(Ke[center_best, :].transpose(1, 0) * scale)
        Z = tmp / tmp.sum(1).unsqueeze(1).repeat(1, K)
        dist = torch.zeros(N, K)#.cuda()

        # run kernel kmeans algorithm (Z[:, k].repeat(N, 1) * Ke).sum(1))
        maxIter_cl = 20
        iter = 0
        while not converged:
            NK = Z.sum(0)
            for k in range(K):
                dist[:, k] = Ke.diag() - (2 / NK[k]) * (Ke.matmul(Z[:,k].diag()).sum(1))  + NK[k].pow(-2) * ((Z[:, k].unsqueeze(1).matmul(Z[:,k].unsqueeze(0))*Ke).sum())

            oldZ = Z
            tmp = torch.exp(-dist * scale)
            Z = tmp / tmp.sum(1).unsqueeze(1).repeat(1, K)

            converged = ((oldZ - Z).abs().sum() < 1)
            if iter >= maxIter_cl:
                converged = 1

            iter = iter + 1

    return Z

def findNeigCluster(Prob_cluster):
    r = 41
    c = 41
    [a, b] = Prob_cluster.max(1)
    lb = b.reshape(r, c)
    Ids = lb.unique()

    # find neighboring labels
    lb_neig = torch.zeros(r*c, 9) #.cuda()
    id = 0
    for px in range(-1, 2):
        for py in range(-1, 2):
            x_list = torch.clamp(torch.tensor(range(px, r + px)), min = 0, max = r - 1)
            y_list = torch.clamp(torch.tensor(range(py, c + py)), min = 0, max = c - 1)
            xs, ys = torch.meshgrid([x_list, y_list])
            lb_shift = lb[xs, ys]
            lb_neig[:, id] = lb_shift.view(-1)
            id = id + 1

    for l in range(Ids.size(0)):
        lb_now = Ids[l]
        mask_idx = (lb.view(-1) == lb_now).nonzero()
        ids_now = lb_neig[mask_idx, :].unique().long()
        list_cluster = ids_now[(ids_now > lb_now)]
        if len(list_cluster.size()):
            list_cluster = torch.cat([lb_now * torch.ones(1, list_cluster.size(0)).long(), list_cluster.unsqueeze(0)], 0)
        else:
            list_cluster = torch.tensor([])

        if l == 0:
            list_cluster_paris = list_cluster
        else:
            list_cluster_paris = torch.cat([list_cluster_paris, list_cluster], 1)

    return list_cluster_paris, lb_neig

def kerKmeans_with_postProcess(pred, pred_raw, feat_hirc, images):

    ncls = 20
    r = 41
    c = 41

    a = time.time()

    ## kernel kmeans algorithm
    Prob_cluster = kernel_kmeans(images, pred, [41, 41], ncls)

    ## remove isolated noisy clusters
    neigClusters, pixel_neigs = findNeigCluster(Prob_cluster) # find neighboring clusters

    #b = time.time()
    #print(b - a)

    numK = Prob_cluster.size(1)
    lb = Prob_cluster.max(1)[1].reshape(r, c)
    for k in range(numK):
        bW =  (lb == k)

        # find connected components
        res = skimage.morphology.label(bW.cpu().numpy())

        # get labels
        unique_labels, inverse, unique_counts = np.unique(res, return_inverse=True, return_counts=True)
        not_background_classes = unique_labels[1:]
        not_background_classes_element_counts = unique_counts[1:]
        #retained_cluster = not_background_classes_element_counts.argmax() + 1

        # set the remaining smaller cluster with labels voted by neighboring clusters
        Nobjs =len(not_background_classes)
        if Nobjs > 1:
            id_retain = not_background_classes_element_counts.argmax()
            for ll in range(0, Nobjs):
                if ll != id_retain:
                    lb_mask = (res.reshape(-1, 1) == not_background_classes[ll]).nonzero()[0] # find the pixels with label of interest

                    # find the list of its neigboring clusters
                    u_lbs, U_inv, u_cots = np.unique(pixel_neigs[lb_mask, :].cpu().numpy(), return_inverse=True, return_counts=True)

                    not_curr_cls = u_lbs[u_lbs != k]
                    not_curr_cots = u_cots[u_lbs != k]

                    # assign label of cluster with maximal number of pixels to the current cluster
                    lb.view(-1)[lb_mask] = int(not_curr_cls[not_curr_cots.argmax()])

    #b = time.time()
    #print(b - a)

    ## build cluster hierachichy by feature similarity
    P = torch.zeros(r * c, numK)
    for ll in range(numK):
        psSet = (lb.view(-1) == ll).nonzero()
        P[psSet, ll] = 1

    P_new = P
    P_list = []

    if 0:
        csize = [feat_hirc[-1].shape[2], feat_hirc[-1].shape[3]]
        interp = nn.Upsample(size=csize, mode='bilinear')
        P_new_up = interp(P_new.reshape(41, 41, -1).permute(2, 0, 1).unsqueeze(0).cuda())
    else:
        Usmap_fun = Upsample_fullScale()
        P_new_up = Usmap_fun(feat_hirc, P_new.reshape(41, 41, -1).permute(2, 0, 1).unsqueeze(0).cuda()).cpu()
    P_list.append(P_new_up)
    #b = time.time()
    #print(b-a)
    for it in range(numK - 1):
        # update the neighboring relationship
        num_cls = P_new.size(1)
        #if it  > 0:
        neigClusters, pixel_neigs = findNeigCluster(P_new)  # find neighboring clusters

        # remove a cluster with smallest distance to neighborling clusters
        n_neigs = neigClusters.size(1)
        list_simi = torch.zeros(n_neigs)
        for l in range(n_neigs):
            ids_clus = neigClusters[:,l]
            ids_clus_neig = (P_new[:, ids_clus[0]] == 1).nonzero()
            ids_clus_curr = (P_new[:, ids_clus[1]] == 1).nonzero()
            xs, ys = torch.meshgrid([ids_clus_neig.squeeze(), ids_clus_curr.squeeze()])

            list_simi[l] = pred_raw[:, xs, ys].mean()

        if list_simi.size(0) > 0:
            ids = list_simi.argmax()

            # merge neigbhoring kernels
            ids_clus = neigClusters[:, ids]
            idx_curr = (P_new[:, ids_clus[1]] == 1).nonzero()
            P_new[idx_curr, ids_clus[1]] = 0
            P_new[idx_curr, ids_clus[0]] = 1
            P_new = P_new[:, (torch.tensor(range(num_cls))!=ids_clus[1].squeeze().cpu()).nonzero().squeeze()]

            # upsampling the clustering results
            if 0:
                P_new_up = interp(P_new.reshape(41, 41, -1).permute(2, 0, 1).unsqueeze(0).cuda())
            else:
                P_new_up = Usmap_fun(feat_hirc, P_new.reshape(41,41,-1).permute(2,0,1).unsqueeze(0).cuda()).cpu()

            P_list.append(P_new_up)

        # = time.time()
        #print(b - a)

    return torch.cat(P_list,1)


#print(P_new.size())
#import matplotlib.pyplot as plt
#nc = prob_clusters_list.size(1)
#plt.imshow(prob_clusters_list[0, nc - 6 : nc - 3, :, :].max[1][1])
#plt.show()


def main():
    """Create the model and start the training."""
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    # Create network.
    #model = Res_Deeplab(num_classes=args.num_classes)
    model = DiffNet_grid()
    #model = ResNet(, num_classes)
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

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
        st = 1 # 310000


    model.train()
    model.cuda()
    
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(BSDataTestSet(args.data_dir, args.data_list, mean=IMG_MEAN))
    if isVOC:
        trainloader = data.DataLoader(VOCDataTestSet(args.data_dir, args.data_list, crop_size=input_size, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=False, num_workers=5, pin_memory=True)
    else:
        trainloader = data.DataLoader(BSDataTestSet(args.data_dir, args.data_list, mean=IMG_MEAN))

    if(0):
        optimizer = optim.Adam([{'params': get_1x_lr_params_NOscale(model), 'lr': 1 * args.learning_rate },
                    {'params': get_10x_lr_params(model), 'lr': 5e-2 * args.learning_rate}],
                    lr=args.learning_rate,  weight_decay=args.weight_decay) #,momentum=args.momentum,

        optimizer.zero_grad()

    interp = nn.Upsample(size=input_size, mode='bilinear')
    neigMask = makeneigMask(41, 41).cuda()

    for i_iter, batch in enumerate(trainloader):
        iter = st + i_iter
        images, fname = batch  #labels,
        images = Variable(images).cuda().permute([0,3,1,2])
        #neigMask = neigMask.cuda()

        #images = images[:,:,:321,:321]

        #  # compute the neightMask online
        r = images.size(2)
        d = images.size(3)
        r = int(r / 8) + 1
        d = int(d / 8) + 1
        #neigMask = makeneigMask(r,d).cuda()

        # apply learned net to image
        a = time.time()
        inputs = (images, neigMask)

        #with torch.no_grad:
        pred, pred_raw, feat_hirc = model(inputs)
        #print(time.time()-a)

        if 1:
            feat = []
            for ll in range(feat_hirc.__len__()):
                feat.append(feat_hirc[ll].detach())

            # upsample the prediction
            prob_clusters_list = kerKmeans_with_postProcess(pred.pow(1).cpu().detach(), pred_raw.cpu().detach(), feat, images.cpu().detach()) #20


        # compute clustering using predicted affinity matrix: to be updated
        #mat_file = scipy.io.loadmat(fname[0])
        #seglabel = mat_file['gt_W'][0] # load g.t. segmentations
        #ncls = 8
        #Prob_cluster = kernel_kmeans(images, pred, [41,41], ncls)

        # print iterations and save the results

        b = time.time()
        print('iter = ', iter, 'of', args.num_steps, 'completed' 'time = ', b - a)
        if isVOC:
            mat_name = [FOLDER_SAVE + fname[0] + '.mat'][0]
        else:
            mat_name = [FOLDER_SAVE + fname[0][42:] + '.mat'][0]  # [42:]

        if 0:
            scipy.io.savemat(mat_name, {'pred': pred.cpu().detach().numpy(),
                                    #'pred_nd': pred_raw.cpu().detach().numpy(),
                                    #'images': images.cpu().detach().numpy(),
                                    #'Prob_cluster': Prob_cluster.cpu().detach().numpy(),
                                    'prob_clusters_list': prob_clusters_list.cpu().detach().numpy(),
                                    'fname': fname[0]})  # {'a_dict': a_dict}

        if 0:
            # To visualize the influence maps
            Usmap_fun = Upsample_fullScale()
            x = 21
            y = 21
            Infl = pred[0, y * 41 + x, :].reshape(1,1,41, 41)

            Infl_up = Usmap_fun(feat_hirc, Infl.cuda()).cpu()
            import matplotlib.pyplot as plt
            plt.imshow(Infl_up.squeeze().cpu().detach())
            plt.show()

            print(fname)

        del pred, inputs

        isdisplay = 0
        if isdisplay == 1:
            import matplotlib.pyplot as plt
            nc = prob_clusters_list.size(1)
            print(nc)
            plt.imshow(prob_clusters_list[0, nc - 6: nc - 3, :, :].max(0)[1])
            plt.show()

        #torch.save(model.state_dict(),osp.join(args.results_dir, 'BSD_diffMap'+str(iter)+'.pth'))

        # save the results into folders

        # what if the input image size is arbitrary?

        # calculate the loss (kernel matching + clustering results)
        # loss_km = loss_calc(pred, gt_Ws)
        # loss_cluster = loss_kerkmeans_ss(pred, gt_clusters)

        #loss = loss_km + loss_cluster

        #loss.backward()
        #optimizer.step()

        #b = time.time()

        #loss_avr = (loss_avr * (i_iter) + loss.data.cpu().numpy()) / (i_iter + 1)

        #print('iter = ', iter, 'of', args.num_steps,'completed, loss = ', [loss_avr, loss.data.cpu().numpy()], 'time = ', b- a )

        #if iter >= args.num_steps-1:
        #    print ('save model ...')
        #    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'BSD_diffMap'+str(args.num_steps)+'.pth'))
        #    break

        #if iter % args.save_pred_every == 0 and i_iter!=0:
        #    print ('taking snapshot ...')
        #    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'BSD_diffMap'+str(iter)+'.pth'))

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

plt.imshow(label_cluster)
plt.show()
'''

'''
import  matplotlib.pyplot as plt
plt.imshow(pred.squeeze().cpu().detach().numpy())
plt.show()
plt.imshow(pred.squeeze()[11 * 61 + 51, :].reshape(r, d).cpu().detach().numpy())
plt.show()

plt.imshow(images[0,0,:,:])
plt.show()
'''

'''
import  matplotlib.pyplot as plt
plt.imshow(pred.squeeze().cpu().detach().numpy())
plt.show()
plt.imshow(pred.squeeze()[20 * 41 + 21, :].reshape(r, d).cpu().detach().numpy())
plt.show()

plt.imshow(images[0,0,:,:])
plt.show()
'''

'''
ncls = 3
Prob_cluster = kernel_kmeans(images, pred, [41, 41], ncls)

import  matplotlib.pyplot as plt
plt.imshow(pred.squeeze().cpu().detach().numpy())
plt.show()
plt.imshow(pred.squeeze()[25 * 41 + 21, :].reshape(41, 41).cpu().detach().numpy())
plt.show()
plt.imshow(Prob_cluster.max(1)[1].squeeze().reshape(41, 41).cpu().detach().numpy())
plt.show()

plt.imshow(images[0,0,:,:])
plt.show()

'''

'''
ncls = 8
Prob_cluster = kernel_kmeans(images, pred, [41, 41], ncls)
import  matplotlib.pyplot as plt
plt.imshow(pred.squeeze().cpu().detach().numpy())
plt.show()
plt.imshow(pred.squeeze()[35 * 41 + 35, :].reshape(41, 41).cpu().detach().numpy())
plt.show()
plt.imshow(Prob_cluster.max(1)[1].squeeze().reshape(41, 41).cpu().detach().numpy())
plt.show()
plt.imshow(images[0,0,:,:])
plt.show()
'''