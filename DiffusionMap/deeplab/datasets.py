import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import scipy.misc
import scipy.io

class VOCDataSet_PseudoLabel_cvpr2018(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, num_cls = 21, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "VOC2012_pseudo_labels_cvpr2018/%s.mat" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape

        # load label
        #label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = scipy.io.loadmat(datafiles["label"]) #, {'pseudo_label': pseudo_label.cpu().detach().numpy(), 'cls_label': cls.cpu().detach().numpy()}) #{'a_dict': a_dict}
        psd_prob = label['pseudo_label']
        #cls_label = label['cls_label']

        psd_prob = cv2.resize(psd_prob, dsize=(size[1], size[0]), interpolation=cv2.INTER_LINEAR)


        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean

        img_h, img_w, n_cls = psd_prob.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(psd_prob, 0, pad_h, 0,
                    pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))

        else:
            img_pad, label_pad = image, psd_prob

        img_h, img_w, n_cls = label_pad.shape
        #n_cls
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w, :], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        #label_full = label.#np.zeros((self.num_cls, self.crop_h, self.crop_w))
        #label_full[cls_label, :, :] = label.transpose((2,0,1))
        label = label.transpose((2,0,1))
        #print(image.shape, label.shape)
        return image.copy(), label.copy(), name #, np.array(cls_label)


class VOCDataSet_PseudoLabel(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, num_cls = 21, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "VOC2012_pseudo_labels/%s.mat" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)

        # load label
        #label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = scipy.io.loadmat(datafiles["label"]) #, {'pseudo_label': pseudo_label.cpu().detach().numpy(), 'cls_label': cls.cpu().detach().numpy()}) #{'a_dict': a_dict}
        psd_prob = label['pseudo_label']
        cls_label = label['cls_label']

        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w, n_cls = psd_prob.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            try:
                label_pad = cv2.copyMakeBorder(psd_prob, 0, pad_h, 0,
                    pad_w, cv2.BORDER_CONSTANT)
            except:
                psd_prob
        else:
            img_pad, label_pad = image, psd_prob

        img_h, img_w, n_cls = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w, :], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        label_full = np.zeros((self.num_cls, self.crop_h, self.crop_w))
        label_full[cls_label, :, :] = label.transpose((2,0,1))

        return image.copy(), label_full.copy(), name #, np.array(cls_label)


class VOCDataSet_PseudoLabel_crop(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, num_cls = 21, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            data_file = osp.join(self.root, "crop_%s.mat" % name)
            #label_file = osp.join(self.root, "VOC2012_pseudo_labels/%s.mat" % name)
            self.files.append({
                "data": data_file,
                #"label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        #fw = 41
        #image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)

        # load label
        #label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        data_load = scipy.io.loadmat(datafiles["data"]) #, {'pseudo_label': pseudo_label.cpu().detach().numpy(), 'cls_label': cls.cpu().detach().numpy()}) #{'a_dict': a_dict}
        ps_label = data_load['prob']
        gt_label = data_load['label']
        image = data_load['image']
        Simi = data_load['Simi']

        image = np.asarray(image, np.float32)
        #img_h, img_w, n_cls = gt_label.shape

        ps_label = np.asarray(ps_label, np.float32)
        Simi = np.asarray(Simi, np.float32)
        gt_label = np.asarray(gt_label, np.float32)
        #image = image.transpose((2, 0, 1))
        #if self.is_mirror:
        #    flip = np.random.choice(2) * 2 - 1
        #    image = image[:, :, ::flip]
        #    ps_label = ps_label[:, ::flip]

        return image.copy(), ps_label.copy(), Simi.copy(), gt_label.copy() #, np.array(cls_label)


class VOCDataSet_PseudoLabel_crop_v2(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, num_cls = 21, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            data_file = osp.join(self.root, "crop_%s.mat" % name)
            #label_file = osp.join(self.root, "VOC2012_pseudo_labels/%s.mat" % name)
            self.files.append({
                "data": data_file,
                #"label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label, pslabel):
        f_scale = 0.9 + random.randint(0, 4) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        pslabel = cv2.resize(pslabel, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label, pslabel

    def __getitem__(self, index):
        datafiles = self.files[index]
        #fw = 41
        #image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)

        ncls = 21

        # load label
        #label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        data_load = scipy.io.loadmat(datafiles["data"]) #, {'pseudo_label': pseudo_label.cpu().detach().numpy(), 'cls_label': cls.cpu().detach().numpy()}) #{'a_dict': a_dict}
        ps_label = torch.from_numpy(data_load['prob']).reshape(ncls, 41, 41).unsqueeze(0)
        gt_label = data_load['label']
        image = data_load['image'].transpose((1,2,0))
        #Simi = data_load['Simi']

        interp = torch.nn.Upsample(size=(self.crop_h, self.crop_w), mode='bilinear')
        ps_label = interp(ps_label).squeeze().permute((1,2,0)).numpy()

        lbs = torch.from_numpy((gt_label)).unique()
        lbs_valid = lbs[lbs != 255]

        #lblist = np.zeros(21)
        #lblist[lbs_valid] = 1

        #print(image.shape, gt_label.shape, ps_label.shape)

        if self.scale:
            image, gt_label, ps_label = self.generate_scale_label(image, gt_label, ps_label)

        image = np.asarray(image, np.float32)
        img_h, img_w, _ = image.shape

        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if ps_label.shape.__len__() == 2:
            ps_label = np.expand_dims(ps_label, axis=2)

        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(gt_label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))


            ps_label_pad = np.zeros((img_h + pad_h, img_w + pad_w, ps_label.shape[2]))

            for p in range(ps_label.shape[2]):
                ps_label_pad[:,:,p] = cv2.copyMakeBorder(ps_label[:,:,p], 0, pad_h, 0,
                                               pad_w, cv2.BORDER_CONSTANT,
                                               value=(0,))
        else:
            img_pad, label_pad, ps_label_pad = image, gt_label, ps_label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        gt_label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        try:
            ps_label = np.asarray(ps_label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w, :], np.float32)
        except:
            print(ps_label.shape, ps_label_pad.shape)

        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
#            label = label[:, ::flip]
            ps_label = ps_label[:, ::flip, :]

        if image.shape[1] != gt_label.shape[0] or image.shape[2] != gt_label.shape[1] or image.shape[1] != self.crop_w or \
                image.shape[2] != self.crop_h:
            print(image.shape)

        #ps_label = np.asarray(ps_label, np.float32)
        #Simi = np.asarray(Simi, np.float32)
        gt_label = np.asarray(gt_label, np.float32)

        ps_label = np.asarray(ps_label, np.float32)

        #lbs_valid = np.asarray(lblist, np.float32)
        #image = image.transpose((2, 0, 1))
        #if self.is_mirror:
        #    flip = np.random.choice(2) * 2 - 1
        #    image = image[:, :, ::flip]
        #    ps_label = ps_label[:, ::flip]

        #ps_label_all = np.zeros((self.crop_h, self.crop_w, ncls))
        #ps_label_all[:, :, lbs_valid] = ps_label

        ps_label_all = np.asarray(ps_label, np.float32).transpose((2,0,1))

        return image.copy(), ps_label_all.copy(), gt_label.copy() #, lbs_valid.copy()#, np.array(cls_label)


class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        #f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = 0.8 + random.randint(0, 4) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
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
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if image.shape[1] != label.shape[0] or image.shape[2] != label.shape[1] or image.shape[1] != self.crop_w or image.shape[2] != self.crop_h:
            print(image.shape)

        return image.copy(), label.copy(), name, np.array(size)


class VOCDataSet_center(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        #f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = 0.8 + random.randint(0, 2) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        image -= self.mean

        if 0:
            img_h, img_w = label.shape
            #f_scale = np.min((1.0, self.crop_w/np.min((img_h, img_w))))
            f_scale = self.crop_w / np.min((img_h, img_w))
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)

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
        h_off = int((img_h - self.crop_h) / 2)
        w_off = int((img_w - self.crop_w) / 2)

        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w])

        self.scale = False
        if self.scale:
            rd = np.random.choice(2) * 2 - 1
            if rd == 1:
                image, label = self.generate_scale_label(image, label)
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

                image, label = img_pad, label_pad

        # randomly process the image data by flipping
        image = image.transpose((2, 0, 1))
        if True:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]


        if image.shape[1] != label.shape[0] or image.shape[2] != label.shape[1] or image.shape[1] != self.crop_w or image.shape[2] != self.crop_h:
            print(image.shape)

        label_onehot = np.zeros((21))
        label_valid = np.unique(label)
        label_onehot[label_valid[label_valid < 255]] = 1

        return image.copy(), label_onehot.copy(), label.copy(), name, np.array(size)


class VOCDataSet_FullImage(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        #f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = 0.8 + random.randint(0, 4) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
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
            # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
            image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
            label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)

        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        #if image.shape[1] != label.shape[0] or image.shape[2] != label.shape[1] or image.shape[1] != self.crop_w or image.shape[2] != self.crop_h:
        #    print(image.shape)

        return image.copy(), label.copy(), name, np.array(size)


class VOCDataSet_diff(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            label_inst_file = osp.join(self.root, "SegmentationClassAug_process/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "label_inst": label_inst_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        #f_scale = 0.5 + random.randint(0, 11) / 10.0
        f_scale = 0.8 + random.randint(0, 4) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label_inst = cv2.imread(datafiles["label_inst"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
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

            label_inst_pad = cv2.copyMakeBorder(label_inst, 0, pad_h, 0,
                   pad_w, cv2.BORDER_CONSTANT,
                   value=(0,)) #self.ignore_label
        else:
            img_pad, label_pad, label_inst_pad = image, label, label_inst

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label_inst = np.asarray(label_inst_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), label_inst.copy(), np.array(size), name

class VOCDataTestSet_MS_CROPS(data.Dataset):
    # crop overlapped fixed patches in (multi-scale, implemented in future version)
    def __init__(self, root, list_path, crop_size=(321, 321), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        #if not max_iters == None:
        #    self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    # implement to get item
    def __getitem__(self, index):

        # multiple scales
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        r, c, nch = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean

        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        #cls_labels = label.unique()

        scales = [0.7, 0.85, 1.0, 1.15, 1.3]
        it = 0
        for id in range(3):
            sc = scales[id]
            im = cv2.resize(image, None, fx=sc, fy=sc, interpolation = cv2.INTER_LINEAR)
            img_h, img_w, _ = im.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                im = cv2.copyMakeBorder(im, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=(0.0, 0.0, 0.0))

            ## croppoing patches
            h_sc, w_sc, nch = im.shape
            h_rg = h_sc - self.crop_h
            w_rg = w_sc - self.crop_w
            npt_h = (np.ceil(h_sc / self.crop_h))
            npt_w = (np.ceil(w_sc / self.crop_w))
            npt = int(npt_h * npt_w)
            if npt_h == 1:
                interval_h = 0
            else:
                interval_h = h_rg / (npt_h - 1)

            if npt_w == 1:
                interval_w = 0
            else:
                interval_w = w_rg / (npt_w - 1)

            ## crop patches from the image
            if id > 0:
                patches = torch.cat((patches, torch.zeros(npt, self.crop_h, self.crop_w, nch)), 0)
                pt = torch.cat((pt, torch.zeros(npt, 5)), 0)
            else:
                patches = torch.zeros(int(npt), self.crop_h, self.crop_w, nch)
                pt = torch.zeros(int(npt), 5)

            for idph in range(int(npt_h)):
                lt_h = int(0 + idph * interval_h)
                for idpw in range(int(npt_w)):
                    lt_w = int(0 + idpw * interval_w)
                    patches[it,:,:,:] = torch.from_numpy(im[lt_h : lt_h + self.crop_h, lt_w : lt_w + self.crop_w, :])
                    pt[it,:] = torch.tensor([lt_h, lt_w, img_h, img_w, sc])
                    it = it + 1

            #image = image.transpose((2, 0, 1))

        return patches.permute(0, 3, 1, 2), pt, image, label, name


class VOCDataTestSet_MS_CROPS_noGT(data.Dataset):
    # crop overlapped fixed patches in (multi-scale, implemented in future version)
    def __init__(self, root, list_path, crop_size=(321, 321), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        #if not max_iters == None:
        #    self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    # implement to get item
    def __getitem__(self, index):

        # multiple scales
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        r, c, nch = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean

        #label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        #cls_labels = label.unique()

        scales = [0.7, 0.85, 1.0, 1.15, 1.3]
        it = 0
        for id in range(3):
            sc = scales[id]
            im = cv2.resize(image, None, fx=sc, fy=sc, interpolation = cv2.INTER_LINEAR)
            img_h, img_w, _ = im.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                im = cv2.copyMakeBorder(im, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=(0.0, 0.0, 0.0))

            ## croppoing patches
            h_sc, w_sc, nch = im.shape
            h_rg = h_sc - self.crop_h
            w_rg = w_sc - self.crop_w
            npt_h = (np.ceil(h_sc / self.crop_h))
            npt_w = (np.ceil(w_sc / self.crop_w))
            npt = int(npt_h * npt_w)
            if npt_h == 1:
                interval_h = 0
            else:
                interval_h = h_rg / (npt_h - 1)

            if npt_w == 1:
                interval_w = 0
            else:
                interval_w = w_rg / (npt_w - 1)

            ## crop patches from the image
            if id > 0:
                patches = torch.cat((patches, torch.zeros(npt, self.crop_h, self.crop_w, nch)), 0)
                pt = torch.cat((pt, torch.zeros(npt, 5)), 0)
            else:
                patches = torch.zeros(int(npt), self.crop_h, self.crop_w, nch)
                pt = torch.zeros(int(npt), 5)

            for idph in range(int(npt_h)):
                lt_h = int(0 + idph * interval_h)
                for idpw in range(int(npt_w)):
                    lt_w = int(0 + idpw * interval_w)
                    patches[it,:,:,:] = torch.from_numpy(im[lt_h : lt_h + self.crop_h, lt_w : lt_w + self.crop_w, :])
                    pt[it,:] = torch.tensor([lt_h, lt_w, img_h, img_w, sc])
                    it = it + 1

            #image = image.transpose((2, 0, 1))

        return patches.permute(0, 3, 1, 2), pt, image, name

class VOCDataTestSet_MS_CROPS_centers(data.Dataset):
    # crop overlapped fixed patches in (multi-scale, implemented in future version)
    def __init__(self, root, list_path, crop_size=(321, 321), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        #if not max_iters == None:
        #    self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    # implement to get item
    def __getitem__(self, index):

        # multiple scales
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        r, c, nch = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean

        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        #cls_labels = label.unique()

        scales = [0.7, 0.85, 1.0, 1.15, 1.3]
        it = 0
        for id in range(4):
            sc = scales[id]
            im = cv2.resize(image, None, fx=sc, fy=sc, interpolation = cv2.INTER_LINEAR)
            img_h, img_w, _ = im.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                im = cv2.copyMakeBorder(im, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=(0.0, 0.0, 0.0))

            ## croppoing patches
            h_sc, w_sc, nch = im.shape
            h_rg = h_sc - self.crop_h
            w_rg = w_sc - self.crop_w
            npt_h = (np.ceil(h_sc / self.crop_h))
            npt_w = (np.ceil(w_sc / self.crop_w))
            npt = int(npt_h * npt_w)
            if npt_h == 1:
                interval_h = 0
            else:
                interval_h = h_rg / (npt_h - 1)

            if npt_w == 1:
                interval_w = 0
            else:
                interval_w = w_rg / (npt_w - 1)

            ## crop patches from the image
            if id > 0:
                patches = torch.cat((patches, torch.zeros(npt, self.crop_h, self.crop_w, nch)), 0)
                pt = torch.cat((pt, torch.zeros(npt, 5)), 0)
            else:
                patches = torch.zeros(int(npt), self.crop_h, self.crop_w, nch)
                pt = torch.zeros(int(npt), 5)

            for idph in range(int(npt_h)):
                lt_h = int(0 + idph * interval_h)
                for idpw in range(int(npt_w)):
                    lt_w = int(0 + idpw * interval_w)
                    patches[it,:,:,:] = torch.from_numpy(im[lt_h : lt_h + self.crop_h, lt_w : lt_w + self.crop_w, :])
                    pt[it,:] = torch.tensor([lt_h, lt_w, img_h, img_w, sc])
                    it = it + 1

            #image = image.transpose((2, 0, 1))

        return patches.permute(0, 3, 1, 2), pt, image, label, name


class VOCDataTestSet_MS_CROPS_v2(data.Dataset):
    # crop overlapped fixed patches in (multi-scale, implemented in future version)
    def __init__(self, root, list_path, crop_size=(321, 321), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        #if not max_iters == None:
        #    self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    # implement to get item
    def __getitem__(self, index):

        # multiple scales
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        r, c, nch = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean

        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        #cls_labels = label.unique()

        img_h, img_w = label.shape
        #f_scale = np.min((1.0, self.crop_w/np.min((img_h, img_w))))
        #f_scale = self.crop_w / np.min((img_h, img_w))
        #image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        #label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(255,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = int((img_h - self.crop_h) / 2)
        w_off = int((img_w - self.crop_w) / 2)

        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w])

        scales = [0.75, 0.8, 0.9, 1.0]
        it = 0
        for id in range(4):
            sc = scales[id]
            im = cv2.resize(image, None, fx=sc, fy=sc, interpolation = cv2.INTER_LINEAR)
            img_h, img_w, _ = im.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                im = cv2.copyMakeBorder(im, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=(0.0, 0.0, 0.0))

            ## croppoing patches
            h_sc, w_sc, nch = im.shape
            h_rg = h_sc - self.crop_h
            w_rg = w_sc - self.crop_w
            npt_h = (np.ceil(h_sc / self.crop_h))
            npt_w = (np.ceil(w_sc / self.crop_w))
            npt = int(npt_h * npt_w)
            if npt_h == 1:
                interval_h = 0
            else:
                interval_h = h_rg / (npt_h - 1)

            if npt_w == 1:
                interval_w = 0
            else:
                interval_w = w_rg / (npt_w - 1)


            ## crop patches from the image
            if id > 0:
                patches = torch.cat((patches, torch.zeros(npt, self.crop_h, self.crop_w, nch)), 0)
                pt = torch.cat((pt, torch.zeros(npt, 5)), 0)
            else:
                patches = torch.zeros(int(npt), self.crop_h, self.crop_w, nch)
                pt = torch.zeros(int(npt), 5)

            for idph in range(int(npt_h)):
                lt_h = int(0 + idph * interval_h)
                for idpw in range(int(npt_w)):
                    lt_w = int(0 + idpw * interval_w)
                    patches[it,:,:,:] = torch.from_numpy(im[lt_h : lt_h + self.crop_h, lt_w : lt_w + self.crop_w, :])
                    pt[it,:] = torch.tensor([lt_h, lt_w, img_h, img_w, sc])
                    it = it + 1

            #image = image.transpose((2, 0, 1))

        return patches.permute(0, 3, 1, 2), pt, image, label, name


class VOCDataTestSet(data.Dataset):
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
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            self.files.append({
                "img": img_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = image[:, :, ::-1]
        image = np.asarray(image, np.float32)
        image -= self.mean
        
        img_h, img_w, _ = image.shape
        self.crop_h = round(img_h/8) * 8 + 1
        self.crop_w = round(img_w/8) * 8 + 1
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
        image = image.transpose((2, 0, 1))
        return image, name #, size


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
