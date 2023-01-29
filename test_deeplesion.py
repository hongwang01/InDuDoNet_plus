import os
import os.path
import argparse
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import h5py
import PIL
from PIL import Image
from network.indudonet_plus import InDuDoNet_plus
from deeplesion.build_gemotry import initialization, build_gemotry
import scipy
from sklearn.cluster import  k_means
import scipy.io as sio
import warnings
warnings.filterwarnings("ignore")


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description="YU_Test")
parser.add_argument("--model_dir", type=str, default="models", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="deeplesion/test/", help='path to training data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--save_path", type=str, default="./test_results/", help='path to training data')
parser.add_argument('--num_channel', type=int, default=32, help='the number of dual channels')
parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='the number of total iterative stages')
parser.add_argument('--eta1', type=float, default=1, help='initialization for stepsize eta1')
parser.add_argument('--eta2', type=float, default=5, help='initialization for stepsize eta2')
parser.add_argument('--alpha', type=float, default=0.5, help='initialization for weight factor')
opt = parser.parse_args()

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")

input_dir = opt.save_path + '/Xma/'
gt_dir = opt.save_path + '/Xgt/'
outX_dir = opt.save_path+'/X/'

mkdir(input_dir)
mkdir(gt_dir)
mkdir(outX_dir)

def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data


def nmarprior(im,threshWater,threshBone,miuAir,miuWater,smFilter):
    imSm = scipy.ndimage.filters.convolve(im, smFilter, mode='nearest')
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

sigma = 1
smFilter =  sio.loadmat('deeplesion/gaussianfilter.mat')['smFilter']

miuAir = 0
miuWater=0.192
starpoint = np.zeros([3, 1])
starpoint[0] = miuAir
starpoint[1] = miuWater
starpoint[2] = 2 * miuWater

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


param = initialization()
ray_trafo = build_gemotry(param)
test_mask = np.load(os.path.join(opt.data_path, 'testmask.npy'))
def test_image(data_path, imag_idx, mask_idx):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    mat_files = open(txtdir, 'r').readlines()
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, 'test_640geo/', data_file)
    gt_absdir = os.path.join(data_path, 'test_640geo/', gt_dir[:-1])
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    Xma= file['ma_CT'][()]
    Sma = file['ma_sinogram'][()]
    XLI = file['LI_CT'][()]
    SLI = file['LI_sinogram'][()]
    Tr = file['metal_trace'][()]
    Sgt = np.asarray(ray_trafo(Xgt))
    file.close()
    M512 = test_mask[:,:,mask_idx]
    M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
    Xprior = nmar_prior(XLI, M)
    Xprior = normalize(Xprior, image_get_minmax())  # *255
    Xma = normalize(Xma, image_get_minmax())  # *255
    Xgt = normalize(Xgt, image_get_minmax())
    XLI = normalize(XLI, image_get_minmax())
    Sma = normalize(Sma, proj_get_minmax())
    Sgt = normalize(Sgt, proj_get_minmax())
    SLI = normalize(SLI, proj_get_minmax())
    Tr = 1 - Tr.astype(np.float32)
    Tr = np.expand_dims(np.transpose(np.expand_dims(Tr, 2), (2, 0, 1)), 0)  # 1*1*h*w
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)),0)
    return torch.Tensor(Xma).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(Xgt).cuda(), torch.Tensor(Mask).cuda(), \
       torch.Tensor(Sma).cuda(), torch.Tensor(SLI).cuda(), torch.Tensor(Sgt).cuda(), torch.Tensor(Tr).cuda(), torch.Tensor(Xprior).cuda()


def print_network(name, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('name={:s}, Total number={:d}'.format(name, num_params))

def main():
    print('Loading model ...\n')
    net = InDuDoNet_plus(opt).cuda()
    print_network("InDuDoNet", net)
    net.load_state_dict(torch.load(opt.model_dir))
    net.eval()
    time_test = 0
    count = 0
    for imag_idx in range(1):  # for demo
        print(imag_idx)
        for mask_idx in range(10):
            Xma, XLI, Xgt, M, Sma, SLI, Sgt, Tr, Xprior = test_image(opt.data_path, imag_idx, mask_idx)
            with torch.no_grad():
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()
                ListX, ListS, ListYS= net(Xma, XLI, Sma, SLI, Tr, Xprior)
            end_time = time.time()
            dur_time = end_time - start_time
            time_test += dur_time
            print('Times: ', dur_time)
            Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
            Xgtclip = torch.clamp(Xgt / 255.0, 0, 0.5)
            Xmaclip = torch.clamp(Xma /255.0, 0, 0.5)
            Xoutnorm = Xoutclip / 0.5
            Xmanorm = Xmaclip / 0.5
            Xgtnorm = Xgtclip / 0.5
            idx = imag_idx *10+ mask_idx  + 1
            plt.imsave(input_dir + str(idx) + '.png', Xmanorm.data.cpu().numpy().squeeze(), cmap="gray")
            plt.imsave(gt_dir + str(idx) + '.png', Xgtnorm.data.cpu().numpy().squeeze(), cmap="gray")
            plt.imsave(outX_dir + str(idx) + '.png', Xoutnorm.data.cpu().numpy().squeeze(), cmap="gray")
            count += 1
    print('Avg.time={:.4f}'.format(time_test/count))
if __name__ == "__main__":
    main()

