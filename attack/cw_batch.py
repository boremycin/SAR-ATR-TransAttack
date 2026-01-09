import torch
import numpy as np
from tqdm import tqdm #type: ignore
import torch.nn as nn
import torch.optim as optim


class CWAttack:
    def __init__(self, model, image=None, c=1, lr=0.001, target=None, max_iteration=100, device=None):
        self.model = model
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if image is not None:  # 单图像模式
            if isinstance(image, np.ndarray):
                self.x = torch.from_numpy(image).float().to(self.device)
            else:
                self.x = image.clone().to(self.device)
            self.x = self.x/255 # only for single image mode 
        self.c = c
        self.lr = lr
        self.target = target
        self.max_iteration = max_iteration
        self.model.eval()

    def attack(self):
        """处理单个图像"""
        images = self.x.unsqueeze(0)  # Add batch dimension
        images = images.to(self.device)
        
        # Initialize w based on the original batch of images
        w = torch.atanh(torch.clamp(2 * images - 1, min=-1 + 1e-6, max=1 - 1e-6)).to(self.device)
        w.requires_grad_(True)

        # Optimizer for the variable w
        optimizer = optim.Adam([w], lr=self.lr)

        # Optimization loop
        for _ in tqdm(range(self.max_iteration)):
            adv_img_raw = (torch.tanh(w) + 1) / 2

            # Calculate the L2 loss (MSE) between the adversarial image and the original image
            loss_mse = nn.MSELoss()(adv_img_raw, images)

            # Calculate the targeted classification loss term
            loss_f_batch = self.loss_f(adv_img_raw)

            # Total loss: L2 loss + c * classification loss
            loss = loss_mse + self.c * loss_f_batch

            # Backpropagate and step the optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Get the final adversarial image batch after optimization
        final_adv_img_raw = (torch.tanh(w) + 1) / 2

        # Clamp the final image to [0, 1] range
        final_adv_img_clamped = torch.clamp(final_adv_img_raw, 0., 1.)

        # Return the adversarial image (remove batch dimension)
        adv_img = final_adv_img_clamped.squeeze(0)
        return adv_img

    def perturb_batch(self, images: torch.Tensor, original_labels: torch.Tensor = None) -> torch.Tensor:
        """处理批量图像"""
        images = images.to(self.device)
        #images = images / 255.0  # normalize to [0, 1]

        # Initialize w based on the original batch of images
        w = torch.atanh(torch.clamp(2 * images - 1, min=-1 + 1e-6, max=1 - 1e-6)).to(self.device)
        w.requires_grad_(True)
        # Optimizer for the variable w
        optimizer = optim.Adam([w], lr=float(self.lr))
        # Optimization loop for the current batch
        for iteration in range(self.max_iteration):
            adv_img_raw = (torch.tanh(w) + 1) / 2
            loss_mse = nn.MSELoss()(adv_img_raw, images)# Calculate the L2 loss (MSE) between the adversarial image and the original image
            loss_f_batch = self.loss_f(adv_img_raw)# Calculate the targeted classification loss term for the batch
            # Total loss: L2 loss + c * classification loss
            loss = loss_mse + self.c * loss_f_batch
            # Backpropagate and step the optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Get the final adversarial image batch after optimization
        final_adv_img_raw = (torch.tanh(w) + 1) / 2
        # Clamp the final image to [0, 1] range
        final_adv_img_clamped = torch.clamp(final_adv_img_raw, 0., 1.)
        return final_adv_img_clamped

    def loss_f(self, adv_batch):
        """
        Calculates the targeted classification loss for a batch of adversarial images.
        """
        results = self.model(adv_batch)  # results shape (N, num_classes)
        # Get logits for the target class for all images in the batch
        target_logits = results[:, self.target]
        # Get max logits for non-target classes for each image in the batch
        mask = torch.ones_like(results, dtype=torch.bool, device=self.device)
        mask[:, self.target] = False
        # Apply mask and reshape
        non_target_logits = results[mask].view(results.shape[0], -1)
        # Find the maximum non-target logit for each image
        max_non_target_logits = torch.max(non_target_logits, dim=1)[0]
        # Calculate the loss for each image
        loss_per_image = torch.clamp_min(max_non_target_logits - target_logits, 0.)
        # The total batch loss is the sum of per-image losses
        total_batch_loss = torch.sum(loss_per_image)
        return total_batch_loss