import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from Options_CDDPE import args


class BraTSDatasetTrain(Dataset):
    def __init__(self, file_dir='./BraTS2018/', transform=transforms.Compose([transforms.ToTensor()])):
        self.T2_file_paths = glob.glob(file_dir + 'T2_2D_train/*.png')
        self.T2_file_paths.sort()
        # self.PD_file_paths.sort()
        self.transform = transform
        self.patch_size = args.patch_size

    def normalizeImg(self, img, low=0, high=1):
        min = np.min(img)
        max = np.max(img)
        k = (high - low) / (max - min + 1e-10)
        imgData = low + k * (img - min)
        imgData[np.isnan(imgData)] = 0
        return imgData

    def get_patch(self, img1, img2):
        h, w = img1.shape[:2]
        stride = self.patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        img1_p = img1[y:y + stride, x:x + stride]
        img2_p = img2[y:y + stride, x:x + stride]

        return img1_p, img2_p

    def __len__(self):
        return len(self.T2_file_paths)

    def __getitem__(self, idx):
        t2_image = Image.open(self.T2_file_paths[idx]).convert('L')
        t1_image = Image.open(self.T2_file_paths[idx].replace(
            "T2_2D_train", "T1_2D_train")).convert('L')

        img_name = self.T2_file_paths[idx].split('/')[-1].replace('.png', '')

        t2_image = np.array(t2_image)
        t1_image = np.array(t1_image)

        img_rec = self.normalizeImg(t2_image)
        img_ref = self.normalizeImg(t1_image)
        if self.transform:
            img_ref = self.transform(img_ref)
            img_rec = self.transform(img_rec)
        return img_ref, img_rec, img_name


class BraTSDatasetTest(Dataset):
    def __init__(self, file_dir='./BraTS2018/', transform=transforms.Compose([transforms.ToTensor()])):
        self.T2_file_paths = glob.glob(file_dir + 'T2_2D_test/*.png')
        self.T2_file_paths.sort()
        self.transform = transform

    def normalizeImg(self, img, low=0, high=1):
        min = np.min(img)
        max = np.max(img)
        k = (high - low) / (max - min + 1e-10)
        imgData = low + k * (img - min)
        imgData[np.isnan(imgData)] = 0
        return imgData

    def __len__(self):
        return len(self.T2_file_paths)

    def __getitem__(self, idx):
        t2_image = Image.open(self.T2_file_paths[idx]).convert('L')
        t1_image = Image.open(self.T2_file_paths[idx].replace(
            "T2_2D_test", "T1_2D_test")).convert('L')

        img_name = self.T2_file_paths[idx].split('/')[-1].replace('.png', '')

        t2_image = np.array(t2_image)
        t1_image = np.array(t1_image)

        img_rec = self.normalizeImg(t2_image)
        img_ref = self.normalizeImg(t1_image)

        if self.transform:
            img_ref = self.transform(img_ref)
            img_rec = self.transform(img_rec)
        return img_ref, img_rec, img_name




class DownUpSample(nn.Module):
    def __init__(self, scale_factor=args.lr_scale, if_up=True):
        super(DownUpSample, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)
        self.if_up = if_up

    def forward(self, x):
        size = x.shape[2:]
        x_down = self.pool(x)
        if self.if_up:
            x_downup = F.interpolate(x_down, size=size, mode='bilinear', align_corners=False)
            return x_downup
        else:
            return x_down

