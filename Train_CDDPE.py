import os
from Options_CDDPE import args
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
from tqdm import trange, tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
from cddpe import CDDPE as Network
from data_brats18 import DownUpSample
from cddpe import loss_fn

if "BraTS" in args.task or "Brats" in args.task or "brats" in args.task:
    from data_brats18 import BraTSDatasetTrain as TrainData

    img_size = 240
elif "ixi" in args.task or "IXI" in args.task:
    from data_ixi import IXIDatasetTrain as TrainData

    img_size = 256

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Mytrain(model_pretrain=None):
    setup_seed(args.seed)
    model_path = f'./modelsave/{args.task}/'
    os.makedirs(model_path, exist_ok=True)

    # device handling
    if args.DEVICE == 'cpu':
        device = 'cpu'
    else:
        device = args.DEVICE
        # device = 'cuda:0'

    model = Network(image_size=img_size)
    if model_pretrain is not None:
        model.load_state_dict(torch.load(model_pretrain, map_location=device))

    model.to(device)
    data_lr_pro = DownUpSample(if_up=True).to(device)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float())])
    train_set = TrainData(transform=transform)
    train_loader = data.DataLoader(train_set,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=1,
                                   pin_memory=True)
    loss_plt = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(0, args.epoch):
        # os.makedirs(args.result + '/' + '%d' % (epoch + 1), exist_ok=True)

        loss_mean = []

        for idx, datas in enumerate(tqdm(train_loader, desc='[Epoch--%d]' % (epoch + 1))):
            # for idx, datas in tqdm(train_loader):
            # print(len(data))
            img_ref_hr, img_hr, _ = datas
            img_lr = data_lr_pro(img_hr)

            img_lr = img_lr.to(device)
            img_ref_hr = img_ref_hr.to(device)
            img_hr = img_hr.to(device)

            img_sr, x_rec, y_rec, x_unique, x_common, y_unique, y_common, warped_y, offset = \
                model(img_lr, img_ref_hr)

            # torch.cuda.empty_cache()
            optimizer.zero_grad()

            loss = loss_fn(img_sr, x_rec, y_rec, x_unique, x_common, y_unique, y_common, warped_y, img_hr,
                           img_ref_hr)

            loss.backward(retain_graph=True)
            optimizer.step()

            loss_mean.append(loss.detach())

        # print loss
        sum_list = 0

        for item in loss_mean:
            sum_list += item

        sum_per_epoch = sum_list / len(loss_mean)

        print('\tLoss:%.5f' % sum_per_epoch)

        loss_plt.append(sum_per_epoch.detach().cpu().numpy())

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epoch:
            torch.save(model.state_dict(), model_path + str(epoch + 1) + '_' + args.task + '.pth')
            print(f'model save in ./modelsave/{args.task}')

    plt.figure()
    x = range(0, args.epoch)
    y = loss_plt
    plt.plot(x, y, 'r-')
    plt.title(args.task, fontsize=16)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(model_path + '/loss.png')


if __name__ == '__main__':
    Mytrain()
