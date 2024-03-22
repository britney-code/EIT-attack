import os
import timm
import torch
import argparse
from torchvision import transforms as T
from torch.utils.data import DataLoader
from loader import ImageNet
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default=r'.\dataset\images.csv',
                    help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default=r'.\dataset\images',
                    help='Input directory with images.')
parser.add_argument('--adv_dir', type=str, default=r"./outputs/", help='Output directory with adversarial images.')
batch_size = 10
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def create_timm_model(model_name):
    model = timm.create_model(model_name, pretrained=True).eval().cuda()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform


def verify(model_name):
    model, transform = create_timm_model(model_name)
    X = ImageNet(opt.adv_dir, opt.input_csv, transform)
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            sum += (model(images).argmax(1) != gt).detach().sum().cpu()
    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))


def main():
    model_names = ["pit_s_224", 'deit_base_patch16_224', 'cait_s24_224']
    for model_name in model_names:
        verify(model_name)
        print("===================================================")


if __name__ == '__main__':
    main()
