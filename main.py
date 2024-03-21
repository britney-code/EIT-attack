import os
import torch
from torch.autograd import Variable as V
from torch import nn
import argparse
from torchvision import transforms as T
from torch.utils.data import DataLoader
from Normalize import Normalize, TfNormalize
from loader import ImageNet
from torch_nets import (
    tf2torch_inception_v3,
    tf2torch_inception_v4,
    tf2torch_resnet_v2_50,
    tf2torch_resnet_v2_101,
    tf2torch_resnet_v2_152,
    tf2torch_inc_res_v2,
    tf2torch_adv_inception_v3,
    tf2torch_ens3_adv_inc_v3,
    tf2torch_ens4_adv_inc_v3,
    tf2torch_ens_adv_inc_res_v2,
)

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default=r'.\dataset\images.csv',
                    help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default=r'.\dataset\images',
                    help='Input directory with images.')
parser.add_argument('--adv_dir', type=str, default="./checkpoint/STM_ens/",
                    help='Output directory with adversarial images.')
batch_size = 10
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf2torch_inception_v3':
        net = tf2torch_inception_v3
    elif net_name == 'tf2torch_inception_v4':
        net = tf2torch_inception_v4
    elif net_name == 'tf2torch_resnet_v2_50':
        net = tf2torch_resnet_v2_50
    elif net_name == 'tf2torch_resnet_v2_101':
        net = tf2torch_resnet_v2_101
    elif net_name == 'tf2torch_resnet_v2_152':
        net = tf2torch_resnet_v2_152
    elif net_name == 'tf2torch_inc_res_v2':
        net = tf2torch_inc_res_v2
    elif net_name == 'tf2torch_adv_inception_v3':
        net = tf2torch_adv_inception_v3
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        net = tf2torch_ens3_adv_inc_v3
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        net = tf2torch_ens4_adv_inc_v3
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf2torch_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    if 'inc' in net_name:
        model = nn.Sequential(
            TfNormalize("tensorflow"),
            net.KitModel(model_path, aux_logits=False).eval().cuda(), )
    else:
        model = nn.Sequential(
            TfNormalize("tensorflow"),
            net.KitModel(model_path).eval().cuda(), )
    return model


def verify(model_name, path):
    model = get_model(model_name, path)
    X = ImageNet(opt.adv_dir, opt.input_csv, T.Compose([T.ToTensor()]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            sum += (model(images).argmax(1) != (gt + 1)).detach().sum().cpu()
    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))


def main():
    model_names = ['tf2torch_inception_v3', 'tf2torch_inception_v4', 'tf2torch_inc_res_v2', 'tf2torch_resnet_v2_152',
                   'tf2torch_resnet_v2_50',
                   'tf2torch_resnet_v2_101', 'tf2torch_adv_inception_v3', 'tf2torch_ens3_adv_inc_v3',
                   'tf2torch_ens4_adv_inc_v3',
                   'tf2torch_ens_adv_inc_res_v2']
    models_path = './torch_nets_weight/'
    for model_name in model_names:
        verify(model_name, models_path)
        print("===================================================")


if __name__ == '__main__':
    main()
