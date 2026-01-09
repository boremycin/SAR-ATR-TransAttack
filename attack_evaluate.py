# adversarial example test
import argparse
import pathlib
import time
import numpy as np
import torch
import torch.nn.functional as F
import tqdm # type: ignore # type: ignore
from pathlib import Path
import sys
import os
from sklearn.metrics import confusion_matrix # type: ignore
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import seaborn as sns # type: ignore
from torchvision import datasets, transforms

# --- Imports for ASR and Confusion Matrix ---
from sklearn.metrics import confusion_matrix # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
# -------------------------------------------
# Assuming these are available from your project structure
from pytorch_image_classification import create_loss # type: ignore
from pytorch_image_classification.utils import ( # type: ignore
    AverageMeter,
    create_logger,
    get_rank,
)
from pytorch_image_classification import (
    get_default_config,
    create_model,
    create_transform,
)
from torchvision import datasets
# Assuming these are your custom helper functions
from attack.attack_utils import load_model_from_pth, load_config # type: ignore


def at_loader(attack_set_path,configs_path,
        batch_size = 8,
        num_workers = 32,
        ):
    """
    attack_set_path：path of adversatial examples file path
    configs_path: as its name 
    """
    config = load_config(configs_path)
    val_transform = create_transform(config, is_train=True)
    dataset = datasets.ImageFolder(root= attack_set_path,transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=None,
                shuffle=False,
                drop_last=False,
                pin_memory=True)
    return test_loader


def evaluate_adversarial_examples(model, dataloader, loss_func, logger, target_label=None, device='cuda:0'):
    """
    Evaluates the model's performance on adversarial examples and calculates ASR and accuracy.

    Args:
        model: The target PyTorch model.
        dataloader: DataLoader providing batches of adversarial images and *original* labels.
                    Images are expected in [0, 1] range and (N, C, H, W) format.
        loss_func: The loss function (e.g., CrossEntropyLoss).
        logger: Logger object.
        target_label: The target class index (integer) used to generate the adversarial examples.
                      If None, only untargeted ASR is calculated.
        device: The device (cpu or gpu) to run the evaluation on.

    Returns:
        preds_np: Concatenated numpy array of raw model outputs (logits).
        probs_np: Concatenated numpy array of prediction probabilities.
        predicted_labels_np: Concatenated numpy array of predicted class indices on adv examples.
        original_labels_np: Concatenated numpy array of original class indices.
        avg_loss: Average loss on the adversarial examples.
        original_acc_on_adv: Standard classification accuracy on adversarial examples
                             (prediction == original label).
        untargeted_asr: Attack success rate (untargeted) = 1 - original_acc_on_adv.
        targeted_asr: Attack success rate (targeted), None if target_label is None.
    """
    device = torch.device(device)
    model.eval()
    loss_meter = AverageMeter()

    # Meters/counters for ASR and Accuracy
    total_images = 0
    total_untargeted_success = 0 # Count where prediction != original label
    total_targeted_success = 0   # Count where prediction == target label
    total_correct_on_adv = 0     # Count where prediction == original label (This is for original accuracy)


    # Lists to collect results for confusion matrix
    original_labels_all = []
    predicted_labels_all = []
    pred_raw_all = []
    pred_prob_all = []


    with torch.no_grad():
        for data, original_targets in tqdm.tqdm(dataloader, desc="Evaluating Adversarial Examples"):
            data = data.to(device)
            original_targets = original_targets.to(device) # Original labels

            outputs = model(data)
            loss = loss_func(outputs, original_targets) # Calculate loss against original labels

            # Get predictions on adversarial examples
            _, predicted_labels = torch.max(outputs, dim=1)

            # --- Calculate Metrics for the batch ---
            num_in_batch = data.size(0)
            total_images += num_in_batch

            # Count correctly classified samples on adversarial examples (Original Accuracy)
            correct_on_adv_batch = predicted_labels.eq(original_targets).sum().item()
            total_correct_on_adv += correct_on_adv_batch

            # Untargeted ASR: prediction is NOT the original label
            # This is equivalent to num_in_batch - correct_on_adv_batch
            untargeted_success_batch = num_in_batch - correct_on_adv_batch
            total_untargeted_success += untargeted_success_batch

            # Targeted ASR: prediction IS the target label (only if target_label is provided)
            if target_label is not None:
                targeted_success_batch = (predicted_labels == target_label).sum().item()
                total_targeted_success += targeted_success_batch

            # --- Collect data for concatenation and confusion matrix ---
            pred_raw_all.append(outputs.cpu().numpy())
            pred_prob_all.append(F.softmax(outputs, dim=1).cpu().numpy())
            predicted_labels_all.append(predicted_labels.cpu().numpy())
            original_labels_all.append(original_targets.cpu().numpy())


            # --- Update Loss Meter ---
            loss_ = loss.item()
            loss_meter.update(loss_, num_in_batch)


    # --- Final Calculations ---
    avg_loss = loss_meter.avg
    original_acc_on_adv = total_correct_on_adv / total_images if total_images > 0 else 0.0
    untargeted_asr = total_untargeted_success / total_images if total_images > 0 else 0.0
    # Note: untargeted_asr should be equal to 1.0 - original_acc_on_adv

    targeted_asr = total_targeted_success / total_images if target_label is not None and total_images > 0 else None


    # Concatenate collected numpy arrays
    preds_np = np.concatenate(pred_raw_all)
    probs_np = np.concatenate(pred_prob_all)
    predicted_labels_np = np.concatenate(predicted_labels_all)
    original_labels_np = np.concatenate(original_labels_all)

    # Log results
    logger.info(f'Evaluation on Adversarial Examples:')
    logger.info(f'Avg Loss: {avg_loss:.4f}')
    logger.info(f'Original Acc (on Adv): {original_acc_on_adv:.4f}')
    #logger.info(f'Untargeted ASR: {untargeted_asr:.4f}') # Should be 1 - Original Acc
    if targeted_asr is not None:
         logger.info(f'Targeted ASR (Target={target_label}): {targeted_asr:.4f}')


    return (preds_np, probs_np, predicted_labels_np,
            original_labels_np, avg_loss, original_acc_on_adv, untargeted_asr, targeted_asr) # Added original_acc_on_adv

def main():
    # --- Configuration Paths ---
    configs_path = "./configs/mstar/pyramidnet.yaml"
    weight_pth = "./weights/mstar/pyramidnet.pth"
    val_save_dir = "./attack_evaluate/transfer-resnext/SRAW2-pyramidnet" # Directory to save evaluation results (npz, confusion matrix)
    AE_dir = "./attack_result/resnext/SRAW2" # Directory containing the generated adversarial examples
    attack_target_class = None # Example target class (replace with your actual target)
    # --- Setup ---
    output_dir = pathlib.Path(val_save_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load config (assuming it contains dataset info like class names)
    config = load_config(configs_path)
    # Setup logger
    logger = create_logger(name=__name__, distributed_rank=get_rank())
    logger.info(f"Evaluating adversarial examples from: {AE_dir}")
    logger.info(f"Saving evaluation results to: {val_save_dir}")
    #if attack_target_class is not None:
        #logger.info(f"Assuming adversarial examples were generated with target class: {attack_target_class}")
    #else:
    logger.info("Evaluating Untargeted ASR.")

    # Load the trained model
    model = load_model_from_pth(configs_path, weight_pth)
    model.eval() # Ensure model is in evaluation mode

    # Create DataLoader for the adversarial examples
    # This loader must provide (adversarial_image_tensor, original_label)
    attack_test_loader = at_loader(AE_dir, configs_path) # Your custom loader
    # Create loss function (used for calculating average loss on adv examples)
    loss_func, _ = create_loss(config) # Assuming create_loss returns loss_func and potentially another value
    # --- Evaluate Adversarial Examples ---
    (preds, probs, predicted_labels,
     original_labels, avg_loss, acc,untargeted_asr, targeted_asr) = evaluate_adversarial_examples(
         model=model,
         dataloader=attack_test_loader,
         loss_func=loss_func,
         logger=logger,
         target_label=attack_target_class, # Pass the target class for targeted ASR
         device=model.fc.weight.device # Use model's device
     )
    # --- Save Evaluation Metrics ---
    output_npz_path = output_dir / f'adversarial_evaluation_results.npz'
    np.savez(
        output_npz_path,
        preds=preds,
        probs=probs,
        predicted_labels=predicted_labels,
        original_labels=original_labels,
        avg_loss=avg_loss,
        original_acc_on_adv=acc, 
        untargeted_asr=untargeted_asr,
        targeted_asr=targeted_asr if targeted_asr is not None else -1 # Save -1 if targeted ASR not calculated
    )
    logger.info(f"Evaluation results saved to {output_npz_path}")

    # --- Generate and Save Confusion Matrix ---
    try:
        # Attempt to get class names from the dataloader's dataset
        class_names = attack_test_loader.dataset.classes if hasattr(attack_test_loader.dataset, 'classes') else [str(i) for i in range(model.fc.out_features)]
    except Exception:
        logger.warning("Could not get class names from dataloader dataset. Using numerical labels.")
        class_names = [str(i) for i in range(model.fc.out_features)] # Fallback to numerical labels

    # Calculate the confusion matrix
    cm = confusion_matrix(original_labels, predicted_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label (on Adversarial Example)')
    plt.ylabel('Original Label')
    plt.title(f'Confusion Matrix on Adversarial Examples\nUntargeted ASR: {untargeted_asr:.4f}' + (f', Targeted ASR (Target={attack_target_class}): {targeted_asr:.4f}' if targeted_asr is not None else ''))
    plt.tight_layout()
    # Save the confusion matrix plot
    output_cm_path = output_dir / f'confusion_matrix_adversarial.png'
    plt.savefig(output_cm_path)
    logger.info(f"Confusion matrix saved to {output_cm_path}")
    plt.close() # Close the plot figure
    logger.info("Adversarial evaluation complete.")

if __name__ == '__main__':
    main()