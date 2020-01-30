import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
import h5py
import scipy.io
from torch.utils import data


class BSDataSet(data.Dataset): #Berkeley segmentation dataset
    def __init__(self, root, list_path, max_iters = None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        #self.img_ids = [self.img_ids[0]] * 10000

        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name)

            self.files.append({
                "img": img_file,
                "name": name
            })
            #self.sample = {'image': image, 'label':label, 'mask':mask} # to be finished!

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.8 + random.randint(0, 4) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST).argmax(2)
        #label = label
        return image, label

    def __getitem__(self, index):
        ######## TO DO LIST: (1) compelete items (2) data argumentation (scaling / flipping)

        #index = 100 # ?????????????????
        datafiles = self.files[index]
        #print(datafiles)

        #mat_file = h5py.File(datafiles["img"])
        mat_file = scipy.io.loadmat(datafiles["img"])
        image = mat_file['img'] #.transpose([2, 0, 1])
        image = image[:, :, ::-1]

        len = mat_file['gt_W'][0].__len__()
        randIdx = torch.LongTensor(1).random_(0, len)

        label = mat_file['gt_W'][0][randIdx]

        if 1:
            lbs = np.unique(label)

            nlbs = lbs.shape[0]
            label_oh = np.zeros([label.shape[0],label.shape[1],nlbs],np.float32)
            for l in range(lbs.shape[0]):
                label_oh[label == lbs[l], l] = 1
            #'''
        if self.scale:
            image, label = self.generate_scale_label(image, label_oh)
        #'''

        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape

        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)

        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        #image = np.asarray(img_pad, np.float32)
        #label = np.asarray(label_pad, np.float32)

        # image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        #'''
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip].copy()
        #''
        gt_lb = np.asanyarray(torch.tensor(label), np.float64)


        resize = torch.nn.Upsample(size = (self.crop_w,self.crop_h), mode = 'bilinear')
        # transform label to hot codes, then upsample the codes
        label = torch.from_numpy(label).long() #.unsqueeze(0).unsqueeze(0).long()


        #ncls = label.max()
        #y_onehot = torch.FloatTensor(label.size(0), ncls, label.size(2), label.size(3)).zero_()
        #y_onehot.scatter_(1, label - 1, 1)
        #label = resize(y_onehot).max(1)[1].squeeze()
        gt_Wei = torch.tensor(label[0::8, 0::8])

        r,d = gt_Wei.size()
        gt_affinity = (gt_Wei.reshape(-1, 1).repeat(1, r * d) == gt_Wei.reshape(1, -1).repeat(r * d, 1))

        gt_affinity = np.asarray(gt_affinity, np.float64)
        gt_Wei = np.asarray(gt_Wei, np.float64)

        return image.copy(), gt_affinity.copy(), gt_lb.copy(), gt_Wei.copy()  #, size, nameneigMask.copy(),

class BSDataSet_nocrop(data.Dataset): #Berkeley segmentation dataset
    def __init__(self, root, list_path, max_iters = None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        #self.img_ids = [self.img_ids[0]] * 10000

        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name)

            self.files.append({
                "img": img_file,
                "name": name
            })
            #self.sample = {'image': image, 'label':label, 'mask':mask} # to be finished!

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.8 + random.randint(0, 4) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST).argmax(2)
        #label = label
        return image, label

    def __getitem__(self, index):
        ######## TO DO LIST: (1) compelete items (2) data argumentation (scaling / flipping)

        #index = 100 # ?????????????????
        datafiles = self.files[index]
        #print(datafiles)

        #mat_file = h5py.File(datafiles["img"])
        mat_file = scipy.io.loadmat(datafiles["img"])
        image = mat_file['img'] #.transpose([2, 0, 1])
        image = image[:, :, ::-1]

        len = mat_file['gt_W'][0].__len__()
        randIdx = torch.LongTensor(1).random_(0, len)

        label = mat_file['gt_W'][0][randIdx]

        if 1:
            lbs = np.unique(label)

            nlbs = lbs.shape[0]
            label_oh = np.zeros([label.shape[0],label.shape[1],nlbs],np.float32)
            for l in range(lbs.shape[0]):
                label_oh[label == lbs[l], l] = 1
            #'''
        if self.scale:
            image, label = self.generate_scale_label(image, label_oh)
        #'''

        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape

        if 0:
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                             pad_w, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                               pad_w, cv2.BORDER_CONSTANT,
                                               value=(self.ignore_label,))
            else:
                img_pad, label_pad = image, label

            img_h, img_w = label_pad.shape

            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)

            image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
            label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

            #image = np.asarray(img_pad, np.float32)
            #label = np.asarray(label_pad, np.float32)

        # image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        #'''
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip].copy()
        #''
        gt_lb = np.asanyarray(torch.tensor(label), np.float64)


        resize = torch.nn.Upsample(size = (self.crop_h,self.crop_w), mode = 'bilinear')
        # transform label to hot codes, then upsample the codes
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).long()

        ncls = label.max() + 1
        y_onehot = torch.FloatTensor(label.size(0), ncls, label.size(2), label.size(3)).zero_()
        y_onehot.scatter_(1, label, 1)
        label = resize(y_onehot).max(1)[1].squeeze()
        gt_Wei = (torch.tensor(label[0::8, 0::8]))

        r,d = gt_Wei.size()
        gt_affinity = (gt_Wei.reshape(-1, 1).repeat(1, r * d) == gt_Wei.reshape(1, -1).repeat(r * d, 1))

        gt_affinity = np.asarray(gt_affinity, np.float64)
        gt_Wei = np.asarray(gt_Wei, np.float64)

        return image.copy(), gt_affinity.copy(), gt_lb.copy(), gt_Wei.copy()  #, size, nameneigMask.copy(),

class BSDataSet_mask(data.Dataset): #Berkeley segmentation dataset
    def __init__(self, root, list_path, max_iters = None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        #self.img_ids = [self.img_ids[0]] * 10000

        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name)

            self.files.append({
                "img": img_file,
                "name": name
            })
            #self.sample = {'image': image, 'label':label, 'mask':mask} # to be finished!

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.8 + random.randint(0, 4) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST).argmax(2)
        #label = label
        return image, label

    def __getitem__(self, index):
        ######## TO DO LIST: (1) compelete items (2) data argumentation (scaling / flipping)

        #index = 100 # ?????????????????
        datafiles = self.files[index]
        #print(datafiles)

        #mat_file = h5py.File(datafiles["img"])
        mat_file = scipy.io.loadmat(datafiles["img"])
        image = mat_file['img'] #.transpose([2, 0, 1])
        image = image[:, :, ::-1]

        len = mat_file['gt_W'][0].__len__()
        randIdx = torch.LongTensor(1).random_(0, len)

        label = mat_file['gt_W'][0][randIdx]

        if 1:
            lbs = np.unique(label)
            ismasked = 0

            nlbs = lbs.shape[0]
            label_oh = np.zeros([label.shape[0],label.shape[1],nlbs],np.float32)
            for l in range(lbs.shape[0]):
                if lbs[l] == -1:
                    ismasked = 1

                label_oh[label == lbs[l], l] = 1
            #'''
        if self.scale:
            image, label = self.generate_scale_label(image, label_oh)
        #'''


        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape

        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)

        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        #image = np.asarray(img_pad, np.float32)
        #label = np.asarray(label_pad, np.float32)

        # image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        #'''
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip].copy()
        #''
        gt_lb = np.asanyarray(torch.tensor(label), np.float64)


        resize = torch.nn.Upsample(size = (self.crop_w,self.crop_h), mode = 'bilinear')
        #transform label to hot codes, then upsample the codes
        label = torch.from_numpy(label).long()#.unsqueeze(0).unsqueeze(0).long()


        #ncls = label.max()
        #y_onehot = torch.FloatTensor(label.size(0), ncls, label.size(2), label.size(3)).zero_()
        #y_onehot.scatter_(1, label - 1, 1)
        #label = resize(y_onehot).max(1)[1].squeeze()
        gt_Wei = torch.tensor(label[0::8, 0::8])


        if ismasked == 1:
            mask = torch.zeros(gt_Wei.size())
            mask[gt_Wei > 0] = 1
            mask[gt_Wei == 255] = 0
        else:
            mask = torch.ones(gt_Wei.size())

        r,d = gt_Wei.size()
        gt_affinity = (gt_Wei.reshape(-1, 1).repeat(1, r * d) == gt_Wei.reshape(1, -1).repeat(r * d, 1))

        gt_affinity = np.asarray(gt_affinity, np.float64)
        gt_Wei = np.asarray(gt_Wei, np.float64)
        mask = np.asarray(mask, np.float64)

        return image.copy(), gt_affinity.copy(), mask.copy(), gt_lb.copy(), gt_Wei.copy()  #, size, nameneigMask.copy(),

class BSDataSet_mask_center(data.Dataset): #Berkeley segmentation dataset
    def __init__(self, root, list_path, max_iters = None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        #self.img_ids = [self.img_ids[0]] * 10000

        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name)

            self.files.append({
                "img": img_file,
                "name": name
            })
            #self.sample = {'image': image, 'label':label, 'mask':mask} # to be finished!

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.8 + random.randint(0, 4) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST).argmax(2)
        #label = label
        return image, label

    def __getitem__(self, index):
        ######## TO DO LIST: (1) compelete items (2) data argumentation (scaling / flipping)

        #index = 100 # ?????????????????
        datafiles = self.files[index]
        #print(datafiles)

        #mat_file = h5py.File(datafiles["img"])
        mat_file = scipy.io.loadmat(datafiles["img"])
        image = mat_file['img'] #.transpose([2, 0, 1])
        image = image[:, :, ::-1]

        len = mat_file['gt_W'][0].__len__()
        randIdx = torch.LongTensor(1).random_(0, len)

        label = mat_file['gt_W'][0][randIdx]

        if 1:
            lbs = np.unique(label)
            ismasked = 0

            nlbs = lbs.shape[0]
            label_oh = np.zeros([label.shape[0],label.shape[1],nlbs],np.float32)
            for l in range(lbs.shape[0]):
                if lbs[l] == -1:
                    ismasked = 1

                label_oh[label == lbs[l], l] = 1
            #'''
        if self.scale:
            image, label = self.generate_scale_label(image, label_oh)
        #'''


        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape

        h_off = int((img_h - self.crop_h) / 2) #random.randint(0, img_h - self.crop_h)
        w_off = int((img_w - self.crop_w) / 2) #random.randint(0, img_w - self.crop_w)

        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        #image = np.asarray(img_pad, np.float32)
        #label = np.asarray(label_pad, np.float32)

        # image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        #'''
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip].copy()
        #''
        gt_lb = np.asanyarray(torch.tensor(label), np.float64)


        resize = torch.nn.Upsample(size = (self.crop_w,self.crop_h), mode = 'bilinear')
        #transform label to hot codes, then upsample the codes
        label = torch.from_numpy(label).long()#.unsqueeze(0).unsqueeze(0).long()


        #ncls = label.max()
        #y_onehot = torch.FloatTensor(label.size(0), ncls, label.size(2), label.size(3)).zero_()
        #y_onehot.scatter_(1, label - 1, 1)
        #label = resize(y_onehot).max(1)[1].squeeze()
        gt_Wei = torch.tensor(label[0::8, 0::8])


        if ismasked == 1:
            mask = torch.zeros(gt_Wei.size())
            mask[gt_Wei > 0] = 1
            mask[gt_Wei == 255] = 0
        else:
            mask = torch.ones(gt_Wei.size())

        r,d = gt_Wei.size()
        gt_affinity = (gt_Wei.reshape(-1, 1).repeat(1, r * d) == gt_Wei.reshape(1, -1).repeat(r * d, 1))

        gt_affinity = np.asarray(gt_affinity, np.float64)
        gt_Wei = np.asarray(gt_Wei, np.float64)
        mask = np.asarray(mask, np.float64)

        return image.copy(), gt_affinity.copy(), mask.copy(), gt_lb.copy(), gt_Wei.copy()  #, size, nameneigMask.copy(),


class BSDataTestSet(data.Dataset):  #Berkeley segmentation dataset
    def __init__(self, root, list_path, crop_size=(505, 505), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = [] 
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "%s" % name)

            self.files.append({
                "img": img_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        ########
        mat_file = scipy.io.loadmat(datafiles["img"])
        image = mat_file['img']
        image = image[:, :, ::-1]
        seglabel = mat_file['gt_W'][0]

        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w, _ = image.shape

        #pad_h = max(self.crop_h - img_h, 0)
        #pad_w = max(self.crop_w - img_w, 0)
        #if pad_h > 0 or pad_w > 0:
        #    image = cv2.copyMakeBorder(image, 0, pad_h, 0,
        #        pad_w, cv2.BORDER_CONSTANT,
        #        value=(0.0, 0.0, 0.0))

        #image = image.transpose((2, 0, 1))
        #image = np.asarray(image, np.float32)
        #seglabel = np.asarray(seglabel, np.float32)
        return image, datafiles["img"] #neigMask, size seglabel,

if __name__ == '__main__':
    dst = VOCDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
