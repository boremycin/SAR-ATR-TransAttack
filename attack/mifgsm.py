from tkinter import TRUE, image_names
from typing_extensions import TypeGuard #type: ignore
from pytorch_image_classification import models
from pytorch_image_classification.utils.dist import get_rank
import torch
import numpy as np
from tqdm import tqdm #type: ignore
import torch.nn as nn
import torch.optim as optim

class MIFGSMAttack:
    def __init__(self, model, image=None, epsilon = 8 / 255, alpha = 2/255, steps = 10 ,target=None, decay = 1.0, random_start=True ,norm = 'linfty',device=None):
        self.model = model
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if image is not None:  # single image mode
            if isinstance(image, np.ndarray):
                self.x = torch.from_numpy(image).float().to(self.device)
            else:
                self.x = image.clone().to(self.device)
            self.x = self.x/255
        self.eps = epsilon
        self.alpha = alpha
        self.steps = steps
        self.target = target
        self.decay = decay
        self.random_start = random_start
        self.norm = norm
        self.model.eval()

    #TODO: To be continued 
    def attack(self):
        """处理单个图像"""
        pass

    def init_delta(self, images, **kwargs):
        delta = torch.zeros_like(images).to(self.device)
        if self.random_start:
            if self.norm == 'linfty':
                delta.uniform_(-self.eps, self.eps)
            else:
                delta.normal_(-self.eps, self.eps)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=-1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(images).uniform_(0,1).to(self.device)
                delta *= r/n*self.eps
            img_min = 0.0  # min value of images
            img_max = 1.0  # max value of images
            delta = torch.clamp(delta, img_min - images, img_max - images)
        delta = delta.detach()
        #delta.requires_grad = True
        return delta
    
    def perturb_batch(self, images: torch.Tensor, original_labels: torch.Tensor = None) -> torch.Tensor:
        """处理批量图像"""
        images = images.to(self.device)
        labels = original_labels.to(self.device)
        outputs = self.model(images)
        if self.target != None:
            target_labels = outputs[:, self.target]

        loss = nn.CrossEntropyLoss()
        delta = self.init_delta(images)
        momentum = torch.zeros_like(images).detach().to(self.device)
        print("start mifgsm iteration")
     
        for _ in tqdm(range(self.steps)):
            adv_images = (images + delta).detach()
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            if self.target != None:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            grad = grad/(torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True) + 1e-12)
            #momentum = self.decay * momentum + grad
            momentum = (self.decay * momentum + grad).detach()
            
            #adv_images = adv_images.detach() + self.alpha * grad.sign()
            if self.norm == 'linfty':
                delta = delta + self.alpha * momentum.sign()
                delta = torch.clamp(delta, -self.eps, self.eps)

            else:
                grad_norm = torch.norm(momentum.view(momentum.size(0), -1), dim=1).view(-1,1,1,1)
                normalized_grad = momentum/(grad_norm + 1e-20)
                delta = delta + self.alpha * normalized_grad
                
                delta_norms = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).view(-1,1,1,1)
                delta = delta * (self.eps/(delta_norms + 1e-12)).clamp(max=1.)
            delta = torch.clamp(images + delta, 0, 1) - images

        adv_images = torch.clamp(images + delta, 0, 1)
        return adv_images   