import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def check_dir_consistency(img_folder: str, mask_folder: str,):
    """
    Checks for consistency between image files and their corresponding mask files in two folders.

    This function verifies that the number of image files in img_folder is equal to the number of mask files 
    in mask_folder. It then checks that each image file has a corresponding mask file with the same name.

    Parameters:
    img_folder (str): Path to the folder containing images.
    mask_folder (str): Path to the folder containing masks.

    Raises:
    AssertionError: If the number of files in img_folder and mask_folder is not equal.
    ValueError: If an image file does not have a corresponding mask file with the same name in mask_folder.

    Returns:
    tuple: A tuple of two lists, the first being the image files and the second being the mask files.
    
    """
    missing_files = False
    img_files = os.listdir(img_folder)
    img_files_no_ext = [img_file.split('.')[0] for img_file in img_files]

    mask_files = os.listdir(mask_folder)
    mask_files_no_ext = [img_file.split('.')[0] for img_file in img_files]

    assert len(img_files) == len(mask_files)
    assert len(img_files_no_ext) == len(mask_files_no_ext)

    # Verify that all images have a corresponding mask
    for img_file in img_files_no_ext:
        if img_file not in mask_files_no_ext:
            print(f"Mask corresponding to image `{img_file}` was not found.")
            missing_files = True

    print(f"Missing files ? {missing_files}")

    return img_files, mask_files


def visualize_images(img_folder: str, mask_folder: str,
                     nb_samples: int = 10, seed: int = 42,
                     mask_postfix: str = 'instance_color_RGB'):
    """
    Visualize random pairs of images and their corresponding masks with common file name as title.

    Parameters:
    img_folder (str): Path to the folder containing images.
    mask_folder (str): Path to the folder containing masks.
    nb_samples (int): Number of random image-mask pairs to display.
    """
    random.seed(seed)
    selection = random.sample(os.listdir(img_folder), nb_samples)

    # Calculate the number of rows needed
    nb_rows = nb_samples // 2 + nb_samples % 2

    plt.figure(figsize=(16, 8))  # Adjusting the overall size of the plot

    # Show images and their corresponding masks
    for i, img_file in enumerate(selection):
        # Load image and mask
        image = Image.open(os.path.join(img_folder, img_file))

        mask_file = img_file.split('.')[0] + '_' + mask_postfix + '.png'
        mask = Image.open(os.path.join(mask_folder, mask_file))

        resolution = f"{image.width}x{image.height}"

        # Set up subplot for image
        plt.subplot(nb_rows, 4, 2 * i + 1)
        plt.imshow(image)
        plt.title(f"{os.path.splitext(img_file)[0]} ({resolution})", fontsize=10)

        plt.axis('off')

        # Set up subplot for mask
        plt.subplot(nb_rows, 4, 2 * i + 2)
        plt.imshow(mask)
        plt.title(f"{os.path.splitext(img_file)[0]} ({resolution})", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def get_all_dims(img_folder: str):
    img_dimensions = []

    for img_file in os.listdir(img_folder):
        with Image.open(os.path.join(img_folder, img_file)) as img:
            img_dimensions.append((img.width, img.height))

    unique_dimensions = set(img_dimensions)
    print(f"Number of unique dimensions in images: {len(unique_dimensions)}")

    return unique_dimensions


def create_histograms_of_dims(img_folder: str, nb_bins: int = None, mult: bool = False):
    """
    Creates histograms of the dimensions of image files in the given folders.

    Parameters:
    img_folder (str): Path to the folder containing images.

    Returns: \nothing
    """
    img_dimensions = []

    if nb_bins is None: nb_bins = int(np.sqrt(len(os.listdir(img_folder))))
    print(f'bin size: {nb_bins}')

    # Collect dimensions of images
    if mult:
        for img_file in os.listdir(img_folder):
            with Image.open(os.path.join(img_folder, img_file)) as img:
                img_dimensions.append(img.width * img.height)
    else:
        for img_file in os.listdir(img_folder):
            with Image.open(os.path.join(img_folder, img_file)) as img:
                img_dimensions.append((img.width, img.height))

    # Plot histogram for image dimensions
    plt.figure(figsize=(24, 6))
    plt.hist(img_dimensions, bins=nb_bins, label='Images')
    plt.xlabel('Dimensions (width x height in pixels)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Image Dimensions')
    plt.legend()
    plt.show()
