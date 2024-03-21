import argparse
import os
import numpy as np
import pretrainedmodels
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from Normalize import Normalize
from loader import ImageNet

parser = argparse.ArgumentParser()
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./checkpoint/outputs_ens/', help='Source Models.')
parser.add_argument("--batch_size", type=int, default=2, help="How many images process at one time.")
opt = parser.parse_args()
transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def save_image(images, names, output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)


if __name__ == '__main__':
    from EIT_Ensemble import EIT
    attack = EIT()
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    from torch_nets import tf2torch_inception_v3
    from torch_nets import tf2torch_inception_v4
    from torch_nets import tf2torch_inc_res_v2
    from torch_nets import tf2torch_resnet_v2_101

    model_path = os.path.join("./torch_nets_weight/", "tf2torch_inception_v3.npy")
    model1 = tf2torch_inception_v3.KitModel(model_path).eval().cuda()
    model_path = os.path.join("./torch_nets_weight/", "tf2torch_inception_v4.npy")
    model2 = tf2torch_inception_v4.KitModel(model_path).eval().cuda()
    model_path = os.path.join("./torch_nets_weight/", "tf2torch_inc_res_v2.npy")
    model3 = tf2torch_inc_res_v2.KitModel(model_path).eval().cuda()
    model_path = os.path.join("./torch_nets_weight/", "tf2torch_resnet_v2_101.npy")
    model4 = tf2torch_resnet_v2_101.KitModel(model_path).eval().cuda()

    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    for images, images_ID, gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)
        adv_img = EIT(images, gt + 1, [model1, model2, model3, model4], images_min, images_max)
        adv_img_np = adv_img.detach().cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)
