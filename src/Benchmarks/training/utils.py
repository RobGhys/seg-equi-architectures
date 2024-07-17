import numpy as np
import torch


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
