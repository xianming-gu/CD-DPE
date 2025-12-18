import numpy as np
import random
import glob
from torch.utils.data import Dataset
from PIL import Image
from Options_CDDPE import args


class IXIDatasetTrain(Dataset):
    def __init__(self, file_dir='./IXI/data/', transform=None):
        self.T2_file_paths = glob.glob(file_dir + 'IXI_T2_2D_train/*.png')
        self.T2_file_paths.sort()
        self.transform = transform
        self.patch_size = args.patch_size

    def normalizeImg(self, img, low=0, high=1):
        min = np.min(img)
        max = np.max(img)
        k = (high - low) / (max - min + 1e-10)
        imgData = low + k * (img - min)
        imgData[np.isnan(imgData)] = 0
        return imgData

    # def loadh5(self, h5_file):
    #     with h5py.File(h5_file, 'r') as f:
    #         # print("Keys: %s" % f.keys())
    #         # kspace_data = f['kspace'][:]
    #         rss_data = f['reconstruction_rss'][:]
    #     return rss_data
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
        pd_image = Image.open(self.T2_file_paths[idx].replace(
            "IXI_T2_2D_train", "IXI_PD_2D_train")).convert('L')

        img_name = self.T2_file_paths[idx].split('/')[-1].replace('.png', '')

        t2_image = np.array(t2_image)
        pd_image = np.array(pd_image)

        img_rec = self.normalizeImg(t2_image)
        img_ref = self.normalizeImg(pd_image)
        if self.transform:
            img_ref = self.transform(img_ref)
            img_rec = self.transform(img_rec)
        return img_ref, img_rec, img_name


class IXIDatasetTest(Dataset):
    def __init__(self, file_dir='./IXI/data/', transform=None):
        self.T2_file_paths = glob.glob(file_dir + 'IXI_T2_2D_test/*.png')
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
        pd_image = Image.open(self.T2_file_paths[idx].replace(
            "IXI_T2_2D_test", "IXI_PD_2D_test")).convert('L')

        img_name = self.T2_file_paths[idx].split('/')[-1].replace('.png', '')

        t2_image = np.array(t2_image)
        pd_image = np.array(pd_image)

        img_rec = self.normalizeImg(t2_image)
        img_ref = self.normalizeImg(pd_image)
        if self.transform:
            img_ref = self.transform(img_ref)
            img_rec = self.transform(img_rec)
        return img_ref, img_rec, img_name


