# grad_cam_visualize.py
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt #type: ignore
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(ROOT).relative_to(Path.cwd())

from pytorch_image_classification import (
    create_model,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import (
    create_logger,
    get_rank,
)
from fvcore.common.checkpoint import Checkpointer


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.model(x)

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        output = self.forward(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        activations = self.activations.detach()
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        heatmap = torch.relu(heatmap)

        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap.cpu().numpy()

def load_config(config_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT/"configs/mstar/vgg.yaml"), type=str, required=False)
    parser.add_argument('--checkpoint', default=str(ROOT/"weights/mstar/vgg.pth"), type=str)
    parser.add_argument('--image_path', type=str, default='attack_result/RPL4/BRDM_2/HB17013.001.jpeg', help='Path to input image')
    parser.add_argument('--output_dir', default=str(ROOT/"resultFigs/vgg/rpl4"), type=str)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.test.checkpoint = args.checkpoint
    config.freeze()
    
    return config, args


def preprocess_image(image_path, config):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)  #increase the dimension of batch
    return input_tensor, image


def visualize_cam(cam, original_image, alpha=0.4):
    cam_resized = cv2.resize(cam, (original_image.width, original_image.height))
    
    original_np = np.array(original_image)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) #make unimportant pixels blue

    original_darkened = original_np.astype(np.float32) / 255.0 
    heatmap_normalized = heatmap.astype(np.float32) / 255.0

    superimposed_img = heatmap_normalized * alpha + original_darkened * (1 - alpha )
    
    superimposed_img = np.clip(superimposed_img, 0, 1) * 255
    superimposed_img = superimposed_img.astype(np.uint8)
    
    return superimposed_img, cam_resized


def main():
    config, args = load_config()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = create_logger(name=__name__, distributed_rank=get_rank())
    model = create_model(config)

    checkpointer = Checkpointer(model)
    checkpointer.load(config.test.checkpoint)
    
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {config.test.checkpoint}")
    
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layers.append((name, module))
    
    if target_layers:
        target_layer_name, target_layer = target_layers[-1]
        logger.info(f"Target layer: {target_layer_name}")
    else:
        target_layer = list(model.modules())[-1]
        logger.info("Using last layer as target")
    
    grad_cam = GradCAM(model, target_layer)
    
    input_tensor, original_image = preprocess_image(args.image_path, config)
    input_tensor = input_tensor.to(device)
    
    logger.info("Generating Grad-CAM...")
    cam = grad_cam.generate_cam(input_tensor)

    superimposed_img, cam_resized = visualize_cam(cam, original_image)

    image_name = Path(args.image_path).stem
    original_save_path = output_dir / f"{image_name}_original.png"
    cam_save_path = output_dir / f"{image_name}_cam.png"
    overlay_save_path = output_dir / f"{image_name}_overlay.png"
    cam_npy_path = output_dir / f"{image_name}_cam.npy"
    
    original_image.save(original_save_path)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cam_resized, cmap='jet')
    plt.colorbar()
    plt.title('Grad-CAM')
    plt.savefig(cam_save_path)
    plt.close()
    
    overlay_image = Image.fromarray(superimposed_img)
    overlay_image.save(overlay_save_path)
    
    np.save(cam_npy_path, cam_resized)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"- Original image: {original_save_path}")
    logger.info(f"- CAM heatmap: {cam_save_path}")
    logger.info(f"- Overlay image: {overlay_save_path}")
    logger.info(f"- CAM data: {cam_npy_path}")


if __name__ == '__main__':
    main()