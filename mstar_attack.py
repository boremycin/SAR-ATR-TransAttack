from typing_extensions import TypeGuard, Self #type: ignore
from matplotlib.artist import setp  # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm #type: ignore
import os
from torchvision.utils import save_image # Useful for saving tensors as images
import torch.utils.data as data # For type hinting DataLoader
import attack
from attack.attack_utils import load_model_from_pth,get_attack_loader,create_attack_transform, at_loader
from attack.cw_batch import CWAttack 
from attack.fgsm import FGSMAttack
from attack.pgd import PGDAttack
from attack.lora_pgd import LoRa_PGDAttack
from attack.mifgsm import MIFGSMAttack
from attack.decowa import DCWAttack
from attack.sraw import SRAWAttack
import argparse


# Assume device is defined globally or passed
# For example:
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generate_and_save_adv(
                        attacker,
                        model: nn.Module, 
                        dataloader: data.DataLoader, 
                        save_dir: str, 
                        max_images: int = 10000, ):
    """
    Generates adversarial examples using the CW attack for images in a DataLoader
    and saves them categorized by their original class.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Attempt to get class names from the dataloader's dataset
    try:
        class_names = dataloader.dataset.classes
    except AttributeError:
        print("Warning: Dataloader dataset does not have a '.classes' attribute. Using label indices for folders.")
        class_names = [str(i) for i in range(1000)] # Placeholder if classes not available

    count = 0 # Total images processed so far
    # Iterate through batches from the dataloader with a progress bar
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Generating and Saving Adversarial Examples")):
        if count >= max_images:
            break # Stop if we've reached the maximum number of images
        batch_size = images.size(0)
        num_to_process = min(batch_size, max_images - count)   # Determine how many images from the current batch to process
        adv_imgs_batch = attacker.perturb_batch(images[:num_to_process], labels[:num_to_process])
        # --- Save the adversarial images from the batch ---
        for i in range(num_to_process):
            # Get original label and class name for saving
            label = labels[i].item()
            # Handle cases where class_names might be just placeholder indices
            class_name = class_names[label] if label < len(class_names) else f"label_{label}"

            # Create class-specific directory if it doesn't exist
            class_dir = os.path.join(save_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Construct the save path
            # Using the global count + i for unique naming across batches
            save_path = os.path.join(class_dir, f"adv_{count + i:05d}.png") # Using 5 digits for count

            # Save the adversarial image (expected in [0, 1] float range)
            save_image(adv_imgs_batch[i].cpu(), save_path) # Move to CPU before saving

        # Update the total count of images processed
        count += num_to_process

    print(f"\nFinished generating and saving. Saved {count} adversarial images to {save_dir}")

# --- Example Usage (requires setting up a dummy model, dataloader, etc.) ---


def parse_args():
    parser = argparse.ArgumentParser(description='Generate CW Adversarial Examples for MSTAR dataset')
    # Required arguments
    parser.add_argument('--attack_set_path', type=str, required=True, help='Path to the attack dataset')
    parser.add_argument('--configs_path', type=str, required=True, help='Path to the model configuration file')
    parser.add_argument('--weight_pth', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--output_directory', type=str, required=True, help='Directory to save adversarial examples')
    # Attack parameters
    parser.add_argument('--target_class', type=int, default=5, help='Target class for the adversarial attack (default: 5)')
    parser.add_argument('--max_images', type=int, default=300000, help='Maximum number of adversarial images to generate (default: 300000)')
    parser.add_argument('--attack_iterations', type=int, default=300, help='Number of attack iterations (default: 300)')
    # Optional arguments
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset (default: 10)')
    return parser.parse_args()


if __name__ == '__main__':
    num_classes = 10
    attack_set_path = "./datasets/MSTAR/ATTACK"
    configs_path = "./configs/mstar/vgg.yaml"
    weight_pth = "./weights/mstar/vgg.pth"
    output_directory = "./attack_result/vgg/CW8"

    model = load_model_from_pth(configs_path,weight_pth)
    attack_loader = get_attack_loader(attack_set_path,configs_path, batch_size=800)
    imgs, labels = next(iter(attack_loader))
    print(imgs.shape)

    model = model.to(device)
   
    # 3. Set attack parameters
    example_target_class = 5 # Example target class (must be < num_classes)
    max_total_images = 30000 # Limit the total number of images saved for this example
    attack_iterations = 50 # Reduced iterations for faster example
   
    attacker_cw = CWAttack(
        model=model,
        c=0.3,
        lr=0.1,
        target=example_target_class,
        max_iteration=1000,
        device=device
    )

    attacker_fgsm = FGSMAttack(
        model=model,
        epsilon= 22/255,
        #target=example_target_class,
        device=device
    )

    attacker_pgd = PGDAttack(
        model=model,
        epsilon= 24/255,
        alpha=32/255,
        steps=50,
        target=None,
        random_start=True,
        device=device
    )
    attacker_lorapgd = LoRa_PGDAttack(
        model=model,
        epsilon=3/255,
        eps_division=1e-10,
        rank=int(128 * 0.6), #adjust 128 according to the size of test image 
        steps=50,
        target=None,
        random_start=True,
        init='lora',
        device=device
    )
    
    attacker_mifgsm = MIFGSMAttack(
        model=model,
        epsilon=20/255,
        alpha=2/255,
        steps=100,
        decay=1,
        norm='linfty',
        device=device
    )

    attacker_decowa = DCWAttack(
        model = model,
        epsilon= 24/255,
        alpha = 4/255, 
        steps = 50,
        target=None,
        decay=1.,
        mesh_width=5,
        mesh_height=5,
        rho=0.01,
        num_warping=15,
        noise_scale=3,
        random_start=True,
        norm='linfty',
        device=device
    )

    attacker_sraw = SRAWAttack(
        model = model,
        image = None,
        epsilon=8/255,
        alpha=4/255,
        steps=400,
        target=None,
        decay=1.0,
        mesh_width=3,
        mesh_height=3,
        rho=0.01,
        num_warping=20,
        center_noise_scale=0.25,
        edge_noise_scale=0.06,
        random_start=True,
        norm='linfty',
        device=device
    )

    generate_and_save_adv(
        attacker_cw,
        model=model,
        dataloader=attack_loader,
        save_dir=output_directory,
        max_images=max_total_images,
    )

    print(f"Adversarial examples saved in subdirectories within: {output_directory}")
    # You can now inspect the 'cw_adv_examples_categorized' directory

