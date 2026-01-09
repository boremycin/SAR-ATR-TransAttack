from os import device_encoding
from tkinter import image_names
from matplotlib import image #type: ignore
from typing_extensions import TypeGuard #type: ignore
from pytorch_image_classification import models
from pytorch_image_classification.utils.dist import get_rank
import torch
import numpy as np
from tqdm import tqdm #type: ignore
import torch.nn as nn
import torch.optim as optim
from .mifgsm import MIFGSMAttack

def K_matrix(X, Y):
    eps = 1e-9
    D2 = torch.pow(X[:, :, None, :] - Y[:, None, :, :], 2).sum(-1)
    K = D2 * torch.log(D2 + eps)
    return K

def P_matrix(X):
    n, k = X.shape[:2]
    device = X.device
    P = torch.ones(n, k, 3, device=device)
    P[:, :, 1:] = X
    return P

def grid_points_2d(width, height, device):
    xx, yy = torch.meshgrid(
        [torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device)])
    return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2)


def noisy_grid(width, height, noise_map, device):
    """
    Make uniform grid points, and add noise except for edge points.
    """
    grid = grid_points_2d(width, height, device)
    mod = torch.zeros([height, width, 2], device=device)
    mod[1:height - 1, 1:width - 1, :] = noise_map
    return grid + mod.reshape(-1, 2)

class TPS_coeffs(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X, Y):

        n, k = X.shape[:2]  # n = 77, k =2
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device) # [1, 80, 80]
        K = K_matrix(X, X)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        # Q = torch.solve(Z, L)[0]
        Q = torch.linalg.solve(L, Z)
        return Q[:, :k], Q[:, k:]

class TPS(torch.nn.Module):
    def __init__(self, size: tuple = (256, 256), device=None):
        super().__init__()
        h, w = size
        self.size = size
        self.device = device
        self.tps = TPS_coeffs()
        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        self.grid = grid.view(-1, h * w, 2)

    def forward(self, X, Y):
        """Override abstract function."""
        h, w = self.size
        W, A = self.tps(X, Y)  
        U = K_matrix(self.grid, X) 
        P = P_matrix(self.grid)
        grid = P @ A + U @ W
        return grid.view(-1, h, w, 2) 

class SRAWAttack(MIFGSMAttack):
    """
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        mesh_width: the number of the control points
        mesh_height: the number of the control points = 3 * 3 = 9
        center_noise_scale = 0.3
        edge_noise_scale = 0.5
        num_warping: the number of warping transformation samples
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    """
    def __init__(self, 
                 model, 
                 image=None, 
                 epsilon = 8/255, 
                 alpha=1.6/255, 
                 steps = 10 ,
                 target=None,  
                 decay = 1.,
                 mesh_width = 3,
                 mesh_height = 3,
                 rho = 0.01,
                 num_warping=20, 
                 center_noise_scale=0.3,
                 edge_noise_scale=0.5,
                 random_start=True,
                 norm = 'linfty', # choose from linfty and gauss
                 device=None):
        super().__init__(model, image, epsilon, alpha, steps, target, decay, random_start, norm, device) # parameters for mifgsm
        self.num_warping = num_warping
        self.center_noise_scale = center_noise_scale
        self.edge_noise_scale = edge_noise_scale
        self.mesh_width = mesh_width
        self.mesh_height = mesh_height
        self.rho = rho        
        self.model.eval()

    #TODO: To be continued 
    def attack(self):
        """single image mode"""
        pass

    def grid_points_src(self):
        width = self.mesh_width
        height = self.mesh_height
        xs = torch.linspace(-1+2/(2*width), 1-2/(2*width), width, device=self.device)
        ys = torch.linspace(-1+2/(2*height), 1-2/(2*height), height, device=self.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        interior = torch.stack([xx, yy], dim = -1).reshape(-1,2)

        boundary = torch.tensor(
            [[-1.0, -1.0],
             [-1.0, 0.0],
             [-1.0, 1.0],
             [0.0, -1.0],
             [0.0, 1.0],
             [1.0, -1.0],
             [1.0, 0.0],
             [1.0, 1.0]], device=self.device)

        return torch.cat([interior, boundary], dim = 0)

    def grid_points_trg(self, noise_map):
        grid = self.grid_points_src()
        noise_map = noise_map.to(self.device)

        num_fixed = 8
        movable = grid[: -num_fixed] + noise_map.reshape(-1, 2)
        fixed = grid[-num_fixed:]

        return torch.cat([movable, fixed], dim = 0)

    def vwt(self, images, noise_map):
        n, c, h, w = images.size()
        X = self.grid_points_src()
        Y = self.grid_points_trg(noise_map)
        tpsb = TPS(size = (h,w), device=self.device)
        warped_grid_b = tpsb(X[None, ...], Y[None, ...])
        warped_grid_b = warped_grid_b.repeat(images.shape[0], 1, 1, 1)
        vwt_imgs = torch.grid_sampler_2d(images, warped_grid_b, 0, 0, False)
        #torch.nn.functional.grid_sample(images, warped_grid_b, align_corners=True)
        return vwt_imgs
    
    def clamp_map(self, noise_map):
        H = self.mesh_height
        W = self.mesh_width

        cell_w = 2.0 / (W - 1) # range is [-1, 1], sun length is 2
        cell_h = 2.0 / (H - 1)

        base_radius = torch.tensor([cell_w, cell_h], device = self.device)
        center_radius = self.center_noise_scale * base_radius
        edge_radius = self.edge_noise_scale * base_radius

        radius_map = edge_radius.view(1,1,2).expand(H,W,2).clone()

        cy = H // 2
        cx = W // 2
    
        radius_map[cy, cx] = center_radius 
        noise_map = torch.clamp(noise_map, -radius_map, radius_map)
        return noise_map

    def init_noise_map(self, std_scale = 0.15):
        H, W = self.mesh_height, self.mesh_width
        noise_map = torch.randn(H, W, 2, device = self.device) * std_scale 
        cell_w = 2.0 / (W - 1)
        cell_h = 2.0 / (H - 1)
        base_radius = torch.tensor([cell_w, cell_h], device = self.device)

        center_radius = base_radius * self.center_noise_scale
        edge_radius = base_radius * self.edge_noise_scale
        radius_map = edge_radius.view(1,1,2).expand(H,W,2).clone()

        cy = H // 2
        cx = W // 2
        radius_map[cy, cx] = center_radius
        noise_map = noise_map * radius_map
        return noise_map

         
    def perturb_batch(self, images: torch.Tensor, original_labels: torch.Tensor = None) -> torch.Tensor:
        """处理批量图像"""
        images = images.clone().detach().to(self.device)
        labels = original_labels.clone().detach().to(self.device)
        images.requires_grad = True
        outputs = self.model(images)

        if self.target != None:
            target_labels = outputs[:, self.target]

        noise_map_hat = self.init_noise_map(0.05)
        noise_map_hat.requires_grad = True

        adv_imgs = self.vwt(images, noise_map_hat)
        #adv_imgs.requires_grad = True

        loss_f = nn.CrossEntropyLoss()
        momentum = 0
        print('start sraw attack iteration')
        for _ in tqdm(range(self.steps)):
            grads = 0
            for _ in range(self.num_warping):
                adv_imgs = self.vwt(images, noise_map_hat)
                logits = self.model(adv_imgs)
                loss = loss_f(logits, labels)

                grad = torch.autograd.grad(loss, noise_map_hat, retain_graph = False, create_graph = False)[0]
                grads += grad
            grads = grads/self.num_warping
            grad_normed = grads / (grads.abs().mean() + 1e-8)
            momentum = momentum * self.decay + grad_normed

            noise_map_hat = noise_map_hat + self.rho * momentum
            noise_map_hat = self.clamp_map(noise_map_hat).detach()
            noise_map_hat.requires_grad = True            
        adv_imgs = self.vwt(images, noise_map_hat)
        adv_imgs = torch.clamp(adv_imgs, 0., 1.)
        return adv_imgs       