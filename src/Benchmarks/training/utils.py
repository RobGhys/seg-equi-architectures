from typing import List

import numpy as np
import torch
import random
import os
from torchvision.utils import save_image, make_grid
import wandb
import json


def overlay_mask(image, mask, color=(1, 1, 1)):
    """Overlay mask on the image. Mask is expected to be in the same size as the image."""
    # Convert mask to 3-channel
    if mask.size(0) == 1:
        mask = mask.repeat(3, 1, 1)

    # Apply the mask
    overlayed = image.clone()
    for c in range(3):
        overlayed[c][mask[c] > 0.5] = color[c]

    return overlayed


def model_sanity_check(settings: dict, n_params: int, device, model) -> None:
    print("\n * Number of parameters:", n_params)
    # Small test of the model
    x = torch.randn((settings['batch_size'], settings['in_channels'],
                     settings['models']['input_size'],
                     settings['models']['input_size'])).to(device=device)
    print("\n * Sanity check of the model:\n",
          "\tinput shape:", x.shape,
          "\n\toutput shape:", model(x).shape)


def save_image_output(imgs, masks, masks_pred, output_path, epoch, i):
    random_index = random.randint(0, imgs.size(0) - 1)
    # Get the selected image, mask, and predicted mask
    img = imgs[random_index]
    mask = masks[random_index]
    mask_pred = masks_pred[random_index]
    # Overlay the masks on the images
    img_with_mask = overlay_mask(img, mask)
    img_with_pred_mask = overlay_mask(img, mask_pred)
    # Stack the original image, image with true mask, and image with predicted mask
    grid = make_grid([img, img_with_mask, img_with_pred_mask], nrow=3)
    # Save the grid image
    save_image(grid, os.path.join(output_path,
                                  'comparison_epoch_{}_batch_{}_idx_{}.png'.format(epoch, i, random_index)))


def denormalize(image, mean=0.5, std=0.5):
    image = image.clone()
    for c in range(3):
        image[c] = image[c] * std + mean
    return image


def save_multiclass_image_output(imgs, masks, masks_pred, output_path, epoch, i, color_map):
    random_index = random.randint(0, imgs.size(0) - 1)

    img = imgs[random_index]
    mask = masks[random_index]
    # only 1 class per pixel
    mask_pred = masks_pred[random_index].argmax(dim=0)

    img = denormalize(img)

    # Overlay masks on top of the image
    img_with_mask = overlay_multiclass_mask(img, mask, color_map)
    img_with_pred_mask = overlay_multiclass_mask(img, mask_pred, color_map)

    grid = make_grid([img.cpu(), img_with_mask, img_with_pred_mask], nrow=3)

    save_image(grid, os.path.join(output_path,
                                  'comparison_epoch_{}.png'.format(epoch)))


def overlay_multiclass_mask(image, mask, color_map):
    """Superpose un masque multiclass sur l'image. Le masque est attendu en taille 2D (H, W)."""
    overlayed = image.clone().cpu()

    # Convertir le masque en image RGB
    mask_rgb = torch.zeros((3, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
    for class_idx, color in color_map.items():
        for c in range(3):  # pour chaque canal (R, G, B)
            mask_rgb[c][mask == class_idx] = color[c]

    # Normaliser le masque RGB
    mask_rgb = mask_rgb.float() / 255.0

    # Superposer le masque RGB à l'image
    overlayed += mask_rgb

    # S'assurer que les valeurs restent dans les limites valides [0, 1]
    overlayed = torch.clamp(overlayed, 0, 1)

    return overlayed


def launch_weights_and_biases(model: str, dataset: str, settings: dict,
                              fold_nb: int, api_key: str):
    try:
        wandb.login(key=api_key)

        config_data = {
            "model": model,
            "dataset": dataset
        }

        lr = settings["models"]["lr"]
        bs = settings['batch_size']

        tags = [
            f'k_fold_idx_{fold_nb}',
            f'batch_size_{bs}',
            f'lr_{lr}'
        ]

        group_name = f"k__{fold_nb}_lr_{lr}_batch_size_{bs}"

        wandb.init(
            project=f"seg_equi_{model}_{dataset}",
            config=config_data,
            group=group_name,
            tags=tags
        )

    except Exception as e:
        print(f"An error occurred while initializing W&B: {e}")


def stop_weights_and_biases():
    try:
        print('Closing W&B...\n')
        wandb.finish()
        print('...Closed W&B')
    except Exception as e:
        print(f"An error occurred while closing W&B: {e}")
