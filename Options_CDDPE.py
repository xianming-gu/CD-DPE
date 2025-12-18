import argparse

parser = argparse.ArgumentParser(description='MyOption')
# Train args
parser.add_argument('--DEVICE', type=str, default='cuda')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--seed', type=int, default=3407)

parser.add_argument('--task', type=str,
                    default='CDDPE_BraTS_bs4_lr0001_sr4')
parser.add_argument('--lr_scale', type=str, default=4)

args = parser.parse_args()



