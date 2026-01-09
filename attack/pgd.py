from tkinter import image_names
from typing_extensions import TypeGuard #type: ignore
from pytorch_image_classification import models
from pytorch_image_classification.utils.dist import get_rank
import torch
import numpy as np
from tqdm import tqdm #type: ignore
import torch.nn as nn
import torch.optim as optim


class PGDAttack:
    def __init__(self, model, image=None, epsilon = 8 / 255, alpha = 2/255, steps = 10 ,target=None, random_start = True ,device=None):
        self.model = model
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if image is not None:  # 单图像模式
            if isinstance(image, np.ndarray):
                self.x = torch.from_numpy(image).float().to(self.device)
            else:
                self.x = image.clone().to(self.device)
            self.x = self.x/255
        self.eps = epsilon
        self.alpha = alpha
        self.steps = steps
        self.target = target
        self.random_start = random_start
        self.model.eval()
        
    #TODO: To be continued 
    def attack(self):
        """处理单个图像"""
        pass

    def perturb_batch(self, images: torch.Tensor, original_labels: torch.Tensor = None) -> torch.Tensor:
        """处理批量图像"""
        images = images.to(self.device)
        labels = original_labels.to(self.device)
        images.requires_grad = True

        if self.target != None:
            target_labels = outputs[:, self.target]

        loss = nn.CrossEntropyLoss()
        outputs = self.model(images)

        adv_imgs = images.clone().detach()

        if self.random_start:
            adv_imgs = adv_imgs + torch.empty_like(adv_imgs).uniform_(-self.eps, self.eps)
            adv_imgs = torch.clamp(adv_imgs, min=0., max=1.).detach()

        print('start pgd attack iteration')
        for _ in tqdm(range(self.steps)):
            adv_imgs = adv_imgs.to(self.device)
            adv_imgs.requires_grad = True

            outputs = self.model(adv_imgs)

            if self.target:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(
                cost, adv_imgs, retain_graph=False, create_graph=False
            )[0]

            adv_imgs = adv_imgs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_imgs - images, min=-self.eps, max=self.eps)
            adv_imgs = torch.clamp(images+delta, min=0., max=1.).detach().to(self.device)
        return adv_imgs       