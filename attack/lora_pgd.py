from os import device_encoding
from tkinter import image_names
from typing_extensions import TypeGuard #type: ignore
from pytorch_image_classification import models
from pytorch_image_classification.utils.dist import get_rank
import torch
import numpy as np
from tqdm import tqdm #type: ignore
import torch.nn as nn
import torch.optim as optim

class LoRa_PGDAttack:
    def __init__(self, model, image=None, epsilon = 8 / 255, eps_division = 1e-10 ,rank = int(128 * 0.2), steps = 10 ,target=None, random_start = True, init = 'lora', device=None):
        self.model = model
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if image is not None:  # 单图像模式
            if isinstance(image, np.ndarray):
                self.x = torch.from_numpy(image).float().to(self.device)
            else:
                self.x = image.clone().to(self.device)
            self.x = self.x/255
        self.eps = epsilon
        self.eps_for_division = eps_division
        self.rank = rank
        self.steps = steps
        self.target = target
        self.random_start = random_start
        self.init = init
        self.model.eval()
    #TODO: To be continued 
    def attack(self):
        """处理单个图像"""
        pass

    def perturb_batch(self, images: torch.Tensor, original_labels: torch.Tensor = None) -> torch.Tensor:
        """处理批量图像"""
        images = images.to(self.device)
        labels = original_labels.to(self.device)

        if self.target != None:
            target_labels = outputs[:, self.target]

        loss_f = nn.CrossEntropyLoss()
        outputs = self.model(images)

        images.requires_grad = True
        bi_shape = images.shape #batch, 3, n, n

        if self.init == 'lora':
            u_im = torch.randn([bi_shape[0], bi_shape[1], bi_shape[2], self.rank], device=self.device) #batch, 3, n, r
            v_im = torch.randn([bi_shape[0], bi_shape[1], self.rank, bi_shape[3]], device = self.device) #batch, 3, r, n
            #norm_u = torch.norm(u_im.view(bi_shape[0], -1), p=2, dim=1)
            #u_im = (u_im / (norm_u.view(bi_shape[0], 1, 1, 1))).detach()
            #v_im = torch.zeros([bi_shape[0], bi_shape[1], self.rank, bi_shape[3]], device = self.device) #batch, 3, r, n

        print('start lora_pgd iteration')
        for _ in tqdm(range(self.steps)):
            u_im = u_im.detach()
            v_im = v_im.detach()

            u_im.requires_grad = True
            v_im.requires_grad = True

            if self.eps == 0.:
                break
            delta = torch.einsum('bcik,bckj->bcij', u_im, v_im)
            delta_norm = torch.linalg.vector_norm(delta.reshape(bi_shape[0], -1), ord=2, dim=1) + self.eps_for_division
            im_per = torch.clamp(images + self.eps * delta/delta_norm.view(bi_shape[0], 1, 1, 1), min=0., max=1.)

            cur_output = self.model(im_per)
            loss = loss_f(cur_output, labels)
            data_grad = torch.autograd.grad(loss, inputs=[u_im, v_im], retain_graph=False, create_graph=False)

            data_grad_u = data_grad[0].detach()
            data_grad_v = data_grad[1].detach()

            norm_grad_u = torch.linalg.vector_norm(data_grad_u.reshape(bi_shape[0],-1), ord=2, dim=1) + self.eps_for_division
            norm_grad_v = torch.linalg.vector_norm(data_grad_v.reshape(bi_shape[0],-1), ord=2, dim=1) + self.eps_for_division

            u_im = u_im + data_grad_u/(norm_grad_u.view(bi_shape[0],1,1,1))
            v_im = v_im + data_grad_v/(norm_grad_v.view(bi_shape[0],1,1,1))

            # u_im = u_im.detach()
            # v_im = v_im.detach()
        
        delta = torch.einsum('bcik,bckj->bcij', u_im, v_im)
        delta_norm = torch.linalg.vector_norm(delta.detach().reshape(bi_shape[0], -1), ord = 2, dim = 1)
        #delta = self.eps * delta / (delta_norm.view(bi_shape[0], 1, 1, 1) + self.eps_for_division)
        delta = self.eps * delta
        adv_imgs = torch.clamp(images + delta, min=0., max=1.)

        return adv_imgs













       