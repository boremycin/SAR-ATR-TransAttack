from tkinter import image_names
import torch
import numpy as np
from tqdm import tqdm #type: ignore
import torch.nn as nn
import torch.optim as optim


class FGSMAttack:
    def __init__(self, model, image=None, epsilon = 8 / 255, target=None, device=None):
        self.model = model
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if image is not None:  # 单图像模式
            if isinstance(image, np.ndarray):
                self.x = torch.from_numpy(image).float().to(self.device)
            else:
                self.x = image.clone().to(self.device)
            self.x = self.x/255
        self.eps = epsilon
        self.target = target
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

        loss = nn.CrossEntropyLoss()
        outputs = self.model(images)
        # targeted attacking mode
        if self.target != None:
            target_labels = outputs[:, self.target]
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)
        
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_imgs = images + self.eps * grad.sign()
        adv_imgs = torch.clamp(adv_imgs, min=0., max=1.)
        return adv_imgs
