import os
import os.path
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
import PIL.Image as Image
from numpy.random import RandomState
import scipy.io as sio
import PIL
from PIL import Image
from .build_gemotry import initialization, build_gemotry
from sklearn.cluster import k_means
import scipy

param = initialization()
ray_trafo = build_gemotry(param)


def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

# hand-crafted prior image
sigma = 1
smFilter = sio.loadmat('deeplesion/gaussianfilter.mat')['smFilter']
miuAir = 0
miuWater=0.192
starpoint = np.zeros([3, 1])
starpoint[0] = miuAir
starpoint[1] = miuWater
starpoint[2] = 2 * miuWater

def nmarprior(im,threshWater,threshBone,miuAir,miuWater,smFilter):
    imSm = scipy.ndimage.filters.convolve(im, smFilter, mode='nearest')
    # print("imSm, h:, w:", imSm.shape[0], imSm.shape[1]) # imSm, h:, w: 416 416
    priorimgHU = imSm
    priorimgHU[imSm <= threshWater] = miuAir
    h, w = imSm.shape[0], imSm.shape[1]
    priorimgHUvector = np.reshape(priorimgHU, h*w)
    region1_1d = np.where(priorimgHUvector > threshWater)
    region2_1d = np.where(priorimgHUvector < threshBone)
    region_1d = np.intersect1d(region1_1d, region2_1d)
    priorimgHUvector[region_1d] = miuWater
    priorimgHU = np.reshape(priorimgHUvector,(h,w))
    return priorimgHU

def nmar_prior(XLI, M):
    XLI[M == 1] = 0.192
    h, w = XLI.shape[0], XLI.shape[1]
    im1d = XLI.reshape(h * w, 1)
    best_centers, labels, best_inertia = k_means(im1d, n_clusters=3, init=starpoint, max_iter=300)
    threshBone2 = np.min(im1d[labels ==2])
    threshBone2 = np.max([threshBone2, 1.2 * miuWater])
    threshWater2 = np.min(im1d[labels == 1])
    imPriorNMAR = nmarprior(XLI, threshWater2, threshBone2, miuAir, miuWater, smFilter)
    return imPriorNMAR


def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data.astype(np.float32)
    data = data*255.0
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data

class MARTrainDataset(udata.Dataset):
    def __init__(self, dir, patchSize, mask):
        super().__init__()
        self.dir = dir
        self.train_mask = mask
        self.patch_size = patchSize
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.rand_state = RandomState(66)
    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        gt_dir = self.mat_files[idx]
      #  random_mask = random.randint(0, 89)  # include 89
        random_mask = random.randint(0, 9)  # for demo
        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'train_640geo/', data_file)
        gt_absdir = os.path.join(self.dir,'train_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma= file['ma_CT'][()]
        Sma = file['ma_sinogram'][()]
        XLI =file['LI_CT'][()]
        SLI = file['LI_sinogram'][()]
        Tr = file['metal_trace'][()]
        file.close()
        Sgt = np.asarray(ray_trafo(Xgt))
        M512 = self.train_mask[:,:,random_mask]
        M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
        Xprior = nmar_prior(XLI, M)
        Xprior = normalize(Xprior, image_get_minmax())  # *255
        Xma = normalize(Xma, image_get_minmax())
        Xgt = normalize(Xgt, image_get_minmax())
        XLI = normalize(XLI, image_get_minmax())
        Sma = normalize(Sma, proj_get_minmax())
        Sgt = normalize(Sgt, proj_get_minmax())
        SLI = normalize(SLI, proj_get_minmax())
        Tr = 1 -Tr.astype(np.float32)
        Tr = np.transpose(np.expand_dims(Tr, 2), (2, 0, 1))
        Mask = M.astype(np.float32)
        Mask = np.transpose(np.expand_dims(Mask, 2), (2, 0, 1))
        return torch.Tensor(Xma), torch.Tensor(XLI), torch.Tensor(Xgt), torch.Tensor(Mask), \
               torch.Tensor(Sma), torch.Tensor(SLI), torch.Tensor(Sgt), torch.Tensor(Tr), torch.Tensor(Xprior)