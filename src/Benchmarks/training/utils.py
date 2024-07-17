import numpy as np
import torch
import random
import os
from torchvision.utils import save_image, make_grid


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