import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from attack_methods import DI, gkern
from Normalize import Normalize, Re_normalize
__all__ = ['EIT']


class EIT:
    def __init__(
            self,
            a=0.2,
            eps=16 / 255,
            steps=10,
            u=1,
            alpha=1.6 / 255,
            ens=20,
            prob=0.1,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
    ):
        self.a = a
        self.eps = eps
        self.device = device
        self.steps = steps
        self.u = u
        self.alpha = alpha
        self.ens = ens
        self.prob = prob
        self.seed_torch(1234)
        self.trans = Normalize(mean, std) # or torchvision.transforms import Normalize

    def seed_torch(self, seed):
        """Set a random seed to ensure that the results are reproducible"""
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    def RPM(self, img, p):
        # get width and height of the image
        s = img.shape
        b = s[0]
        wd = s[1]
        ht = s[2]
        img_copy = np.copy(img)
        np.random.shuffle(img_copy)
        # possible grid size, 0 means no hiding
        grid_sizes = [0, 20, 40, 60, 80]
        # hiding probability
        hide_prob = p
        # randomly choose one grid size
        grid_size = grid_sizes[random.randint(0, len(grid_sizes) - 1)]
        # hide the patches
        if grid_size != 0:
            for x in range(0, wd, grid_size):
                for y in range(0, ht, grid_size):
                    x_end = min(wd, x + grid_size)
                    y_end = min(ht, y + grid_size)
                    if random.random() <= hide_prob:
                        img[:, x:x_end, y:y_end, :] = np.random.uniform(low=0, high=1, size=np.shape(img[:, x:x_end, y:y_end, :]))
        return img

    def __call__(self, model: nn.Module, inputs: torch.tensor, labels: torch.tensor, **kwargs):
        if torch.max(inputs) > 1 or torch.min(inputs) < 0:
            raise ValueError('Input must have a range [0, 1] (max: {}, min: {})'.format(
                torch.max(inputs), torch.min(inputs))
            )
        g = torch.zeros_like(inputs)
        images = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv = images.clone().detach()
        for i in range(self.steps):
            temp_weight = 0
            for l in range(self.ens):
                temp = adv.clone().detach()
                temp = temp.cpu().numpy().transpose(0, 2, 3, 1)
                images_tmp = self.RPM(temp, self.prob)
                images_tmp = torch.from_numpy(images_tmp).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)
                images_tmp = self.trans(images_tmp)             # images_tmp = images_tmp * 2.0 - 1.0
                images_tmp += torch.from_numpy(np.random.uniform(low=-self.a, high=self.a, size=images_tmp.shape)).to(self.device)
                images_tmp = images_tmp * (1 - l / self.ens)
                images_tmp.requires_grad = True
                logits = model(images_tmp)
                logits = F.softmax(logits, 1)
                labels_onehot = F.one_hot(labels, len(logits[0])).float()
                score = logits * labels_onehot
                loss = torch.sum(score)
                adv_grad = torch.autograd.grad(loss, images_tmp, retain_graph=False, create_graph=False)[0]
                temp_weight += adv_grad

            grad = temp_weight / self.ens
            g = self.u * g + grad / (torch.mean(torch.abs(grad), [1, 2, 3], keepdim=True))
            adv = adv - self.alpha * torch.sign(g)
            delta = torch.clip(adv - inputs, -self.eps, self.eps)
            adv = torch.clip(inputs + delta, 0, 1).detach_()
        return adv
