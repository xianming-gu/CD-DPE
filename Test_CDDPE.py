import os
from Options_CDDPE import args
import numpy as np
import torch
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from data_brats18 import DownUpSample
import torch.utils.data as data
import torchvision.transforms as transforms
from cddpe import CDDPE as Network

if "BraTS" in args.task or "Brats" in args.task or "brats" in args.task:
    from data_brats18 import BraTSDatasetTest as TestData

    img_size = 240
elif "ixi" in args.task or "IXI" in args.task:
    from data_ixi import IXIDatasetTest as TestData

    img_size = 256

EPS = 1e-8


def Mytest(model_test=None, img_save_dir=None):
    if model_test is None:
        model_path_final = './modelsave/' + args.task + '/' + str(args.epoch) + '_' + args.task + '.pth'
    else:
        model_path_final = model_test

    if img_save_dir is None:
        img_save_dir = './result/' + args.task
    else:
        img_save_dir = img_save_dir
    # device handling
    if args.DEVICE == 'cpu':
        device = 'cpu'
    else:
        device = args.DEVICE

    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(f'{img_save_dir}/SR/', exist_ok=True)

    model = Network(image_size=img_size)
    model.eval()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path_final, map_location=device))
    data_lr_pro = DownUpSample(scale_factor=args.lr_scale, if_up=True).to(device)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float())])
    test_set = TestData(transform=transform)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False,
                                  num_workers=1, pin_memory=False)
    psnr_list = []
    ssim_list = []
    with torch.no_grad():
        for batch, datas in enumerate(test_loader):
            img_ref_hr, img_hr, img_name = datas

            print("test for image %s" % img_name[0], end='\t')

            img_lr = data_lr_pro(img_hr)
            img_lr = img_lr.to(device)
            img_ref_hr = img_ref_hr.to(device)

            img_sr, *_ = model(img_lr, img_ref_hr)

            img_sr = (img_sr - img_sr.min()) / (img_sr.max() - img_sr.min()) * 255.
            img_sr = img_sr.cpu().numpy().squeeze()
            img_hr = img_hr.cpu().numpy().squeeze()

            psnr_per_img, ssim_per_img = evaluate_super_resolution(img_sr, img_hr)
            print(f'psnr:{psnr_per_img}, ssim:{ssim_per_img}')
            psnr_list.append(psnr_per_img)
            ssim_list.append(ssim_per_img)
            cv2.imwrite(f'{img_save_dir}/SR/{img_name[0]}.png', img_sr)
    print('test results in %s/' % img_save_dir)
    with open(f'{img_save_dir}/metrics.txt', 'a') as file:
        file.write(f'{args.task}\n')
        file.write(f'PSNR_avg:\t{round(sum(psnr_list) / len(psnr_list), 5)}\n')
        file.write(f'SSIM_avg:\t{round(sum(ssim_list) / len(ssim_list), 5)}')
    print(f'PSNR_avg:\t{round(sum(psnr_list) / len(psnr_list), 5)}')
    print(f'SSIM_avg:\t{round(sum(ssim_list) / len(ssim_list), 5)}')
    print('Finish!')


def evaluate_super_resolution(sr_image, hr_image):
    sr_image = (sr_image - sr_image.min()) / (sr_image.max() - sr_image.min())
    hr_image = (hr_image - hr_image.min()) / (hr_image.max() - hr_image.min())
    sr_array = np.array(sr_image)
    hr_array = np.array(hr_image)

    psnr_val = psnr(hr_array, sr_array, data_range=1.0)
    ssim_val = ssim(hr_array, sr_array, data_range=1.0)

    return psnr_val, ssim_val


if __name__ == '__main__':
    Mytest()
