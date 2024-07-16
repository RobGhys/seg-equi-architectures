from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

from PIL import Image
import os

import torch
from torchvision.utils import make_grid, save_image


class BasicDataset(Dataset):
    """
    Images are supposed to be RGB
    Masks are supposed to be L (binary grayscale)
    """

    def __init__(self, dataset_path, filenames=[], mask_type='single_class',
                 transforms={'Resize': None, 'RandomCrop': None, 'flip': True, 'rot': True,
                             'Normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))}):
        self.dataset_path = dataset_path
        self.filenames = filenames
        self.mask_type = mask_type
        self.transforms = transforms

        if isinstance(self.transforms['Resize'], list):
            self.transforms['Resize'] = tuple(self.transforms['Resize'])

        print(f'BasicDataset transforms : {self.transforms}')

    def __getitem__(self, index, debug=False):
        img_path = os.path.join(self.dataset_path, 'imgs', self.filenames[index])
        mask_path = os.path.join(self.dataset_path, 'masks', self.filenames[index])
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # elif self.mask_type == 'multiclass_semantic':
        #     raise NotImplementedError("Not implemented yet")
        # elif self.mask_type == 'multiclass_instance':
        #     raise NotImplementedError("Not implemented yet")
        # else:
        #     raise ValueError(
        #         "Unknown mask type: should be 'single_class', 'multiclass_semantic' or 'multiclass_instance'")

        if debug:
            print(f"Original img type: {type(img)}, size: {img.size}")
            print(f"Original mask type: {type(mask)}, size: {mask.size}")

        if self.transforms['RandomCrop']:
            i, j, h, w = transforms.RandomCrop.get_params(img, self.transforms['RandomCrop'])
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
            if debug:
                print(f"RandomCrop: img size {img.size}, mask size {mask.size}")

        if self.transforms['flip']:
            if np.random.rand() > 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)
            if np.random.rand() > 0.5:
                img = F.vflip(img)
                mask = F.vflip(mask)

        if self.transforms['rot']:
            d = np.random.choice([0, 90, 180, 270])
            if d != 0:
                img = F.rotate(img, int(d))
                mask = F.rotate(mask, int(d))

        if self.transforms['Resize']:
            img = F.resize(img, self.transforms['Resize'])
            mask = F.resize(mask, self.transforms['Resize'])
            if debug:
                print(f"Resize: img size {img.size}, mask size {mask.size}")

        tf = [transforms.ToTensor()]
        if self.transforms['Normalize']:
            tf.append(transforms.Normalize(*self.transforms['Normalize']))
        tf_im = transforms.Compose(tf)
        tf_ma = transforms.Compose(tf[:-1])

        img = tf_im(img)
        mask = tf_ma(mask)

        if debug:
            print(f"Final tensor types and shapes: img {type(img)}, shape {img.shape}, mask {type(mask)}, shape {mask.shape}")

        return img, mask

    def __len__(self):
        return len(self.filenames)

def custom_collate_fn(batch):
    imgs, masks = zip(*batch)
    print(f"Collating batch: {[type(img) for img in imgs]}")
    for i, img in enumerate(imgs):
        print(f"  Batch element {i} type: {type(img)}, shape: {img.shape if isinstance(img, torch.Tensor) else 'N/A'}")
    for i, mask in enumerate(masks):
        print(f"  Batch element {i} type: {type(mask)}, shape: {mask.shape if isinstance(mask, torch.Tensor) else 'N/A'}")
    return torch.stack(imgs, 0), torch.stack(masks, 0)


def getDataLoader(settings, fold):
    path = settings['path']

    # Get the folds that will be used for training and testing, respectively
    folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
    testing_folds = [folds.pop(fold)]
    training_folds = folds

    # Get the filenames of the images in the training and testing folds
    training_data = []
    for fold in training_folds:
        training_data += [os.path.join(fold, f) for f in os.listdir(os.path.join(path, 'imgs', fold))
                          if os.path.isfile(os.path.join(path, 'imgs', fold, f))]

    testing_data = []
    for fold in testing_folds:
        testing_data += [os.path.join(fold, f) for f in os.listdir(os.path.join(path, 'imgs', fold))
                         if os.path.isfile(os.path.join(path, 'imgs', fold, f))]

    # Create the datasets
    train_data = BasicDataset(path, training_data,
                              mask_type=settings['mask_type'], transforms=settings['transforms'])
    test_data = BasicDataset(path, testing_data,
                             mask_type=settings['mask_type'], transforms=settings['transforms'])

    # Create the dataloaders
    train_loader = DataLoader(train_data, batch_size=settings['batch_size'],
                              shuffle=settings['shuffle'], num_workers=settings['num_workers'])
                              #collate_fn=custom_collate_fn)

    test_loader = DataLoader(test_data, batch_size=settings['batch_size'],
                             shuffle=settings['shuffle'], num_workers=settings['num_workers'])
                             #collate_fn=custom_collate_fn)

    return train_loader, test_loader


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def visualize_batch(images: torch.Tensor, masks: torch.Tensor, output_path: str,
                    epoch: int, num_images: int = 3,
                    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)) -> None:
    batch_size = images.size(0)
    # Randomly get 'num_images'
    indices = torch.randperm(batch_size)[:num_images]

    selected_imgs = images[indices]
    selected_masks = masks[indices]

    combined_images: List[torch.Tensor] = []

    for i in range(num_images):
        img = selected_imgs[i]
        mask = selected_masks[i]

        # Remove normalization for better visualisation
        img_denorm = denormalize(img.clone(), mean, std)
        combined_images.append(img_denorm)

        img_with_mask = img_denorm.clone()
        img_with_mask[:, mask[0] > 0.5] = 1  # Supersede a white mask on top of the image
        combined_images.append(img_with_mask)

    # Create a Tensor from the list
    combined_images = torch.stack(combined_images)

    grid = make_grid(combined_images, nrow=2)
    save_image(grid, os.path.join(output_path, 'visualization_epoch_{}.png'.format(epoch)))


if __name__ == "__main__":
    settings = {
        'path': "/home/rob/Documents/3_projects/bench/kvasir",
        'mask_type': 'single_class',
        'transforms': {
            'Resize': (128, 128),
            'RandomCrop': None,
            'flip': False,
            'rot': False,
            'Normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        },
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 2
    }
    fold = 0
    output_path = os.path.join(os.getcwd(), 'outputs')
    epoch = 1

    train_loader, test_loader = getDataLoader(settings, fold)

    for images, masks in train_loader:
        visualize_batch(images, masks, output_path, epoch, num_images=3, mean=settings['transforms']['Normalize'][0],
                        std=settings['transforms']['Normalize'][1])
        break
