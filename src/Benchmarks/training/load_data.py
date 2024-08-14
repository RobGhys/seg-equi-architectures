import json
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from typing import Union
import random

from src.Benchmarks.training.CocoStuffDataset import CocoStuffDataset


class BasicDataset(Dataset):
    """
    Images are supposed to be RGB
    Masks are supposed to be L (binary grayscale) or RGB (multiclass)
    """

    def __init__(self, dataset_path, filenames=[], mask_type='single_class',
                 transforms={'Resize': None, 'RandomCrop': None, 'flip': True, 'rot': True,
                             'Normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))},
                 multiclass_palette_path=None):
        self.dataset_path = dataset_path
        self.filenames = sorted(filenames)
        self.mask_type = mask_type
        self.transforms = transforms
        self.multiclass_palette = None
        self.palette = None

        if isinstance(self.transforms['Resize'], list):
            self.transforms['Resize'] = tuple(self.transforms['Resize'])

        if multiclass_palette_path:
            with open(multiclass_palette_path, 'r') as f:
                self.multiclass_palette = json.load(f)
            self.palette = {tuple(v): int(k) for k, v in self.multiclass_palette.items()}

        print(f'BasicDataset transforms : {self.transforms}')

    def convert_rgb_to_class_indices(self, mask, debug=False) -> torch.Tensor:
        # Convert RGB mask to class indices
        mask_np = np.array(mask)
        class_indices = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)

        if debug:
            print(f'shape of mask_np: {mask_np.shape}')
            unique_colors = np.unique(mask_np.reshape(-1, mask_np.shape[2]), axis=0)
            print(f'unique values in mask_np: {unique_colors}')

        for rgb, idx in self.palette.items():
            matches = np.all(mask_np == np.array(rgb).reshape(1, 1, 3), axis=-1)
            class_indices[matches] = idx

            if debug:
                print(f'Checking for RGB value: {rgb}, Index: {idx}, Matches found: {np.sum(matches)}')

        if debug:
            print(f'unique values in class_indices: {np.unique(class_indices)}')

        class_indices = torch.as_tensor(class_indices, dtype=torch.long).unsqueeze(0)  # Shape [1, H, W]

        return class_indices

    def __getitem__(self, index, debug=False):
        img_path = os.path.join(self.dataset_path, 'imgs', self.filenames[index])
        mask_path = os.path.join(self.dataset_path, 'masks', self.filenames[index])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transforms['RandomCrop']:
            i, j, h, w = transforms.RandomCrop.get_params(img, self.transforms['RandomCrop'])
            img: Union[Image.Image, torch.Tensor] = F.crop(img, i, j, h, w)
            mask: Union[Image.Image, torch.Tensor] = F.crop(mask, i, j, h, w)
            if debug:
                print(f"RandomCrop: img size {img.size}, mask size {mask.size}")

        if self.transforms['flip']:
            if np.random.rand() > 0.5:
                img: Union[Image.Image, torch.Tensor] = F.hflip(img)
                mask: Union[Image.Image, torch.Tensor] = F.hflip(mask)
            if np.random.rand() > 0.5:
                img: Union[Image.Image, torch.Tensor] = F.vflip(img)
                mask: Union[Image.Image, torch.Tensor] = F.vflip(mask)

        if self.transforms['rot']:
            d = np.random.choice([0, 90, 180, 270])
            if d != 0:
                img: Union[Image.Image, torch.Tensor] = F.rotate(img, int(d))
                mask: Union[Image.Image, torch.Tensor] = F.rotate(mask, int(d))

        if self.transforms['Resize']:
            img: Union[Image.Image, torch.Tensor] = F.resize(img, self.transforms['Resize'])
            mask: Union[Image.Image, torch.Tensor] = F.resize(mask, self.transforms['Resize'],
                                                              interpolation=Image.NEAREST)
            if debug:
                print(f"Resize: img size {img.size}, mask size {mask.size}")
                print(f'type of img: {type(img)} | type of mask: {type(mask)}')

        img: torch.Tensor = transforms.ToTensor()(img)
        if self.transforms['Normalize']:
            img = transforms.Normalize(*self.transforms['Normalize'])(img)

        if self.mask_type == 'multiclass_semantic':
            mask: torch.Tensor = self.convert_rgb_to_class_indices(mask)
        elif self.mask_type == 'single_class':
            mask = mask.convert('L')
            mask: torch.Tensor = transforms.ToTensor()(mask)
            mask: torch.Tensor = (mask > 0).long()

        if self.transforms['Normalize']:
            img = transforms.Normalize(*self.transforms['Normalize'])(img)

        if debug:
            non_zero_elements = mask[mask != 0]

            print(f'Count of Non-zero elements in mask: {len(non_zero_elements)}')
            print(
                f"Final tensor types and shapes: img -> {type(img)}, shape {img.shape}, mask -> {type(mask)}, shape {mask.shape}")
            print(f'example mask: {mask}')

        data = {'img': img, 'mask': mask, 'img_path': img_path, 'mask_path': mask_path}

        #return img, mask, img_path, mask_path
        return data

    def __len__(self):
        return len(self.filenames)


def get_data_loader(settings, fold, subset_data: bool = False):
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

    if settings['name'] != 'coco':
        # Create the datasets
        train_data = BasicDataset(path,
                                  training_data,
                                  mask_type=settings['mask_type'],
                                  transforms=settings['transforms'],
                                  multiclass_palette_path=settings['multiclass_palette_path'])
        test_data = BasicDataset(path,
                                 testing_data,
                                 mask_type=settings['mask_type'],
                                 transforms=settings['transforms'],
                                 multiclass_palette_path=settings['multiclass_palette_path'])

        if subset_data:
            subset_train_size = 5000
            indices_train = list(range(len(train_data)))
            subset_train_indices = indices_train[:subset_train_size]

            train_sampler = SubsetRandomSampler(subset_train_indices)
            train_loader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=settings['batch_size'],
                                      num_workers=settings['num_workers'],
                                      pin_memory=True)

            subset_test_size = 800
            indices_test = list(range(len(test_data)))
            subset_test_indices = indices_test[:subset_test_size]

            test_sampler = SubsetRandomSampler(subset_test_indices)
            test_loader = DataLoader(test_data,
                                     sampler=test_sampler,
                                     batch_size=settings['batch_size'],
                                     num_workers=settings['num_workers'],
                                     pin_memory=True)
        else:
            # Create the dataloaders
            train_loader = DataLoader(train_data,
                                      batch_size=settings['batch_size'],
                                      shuffle=settings['shuffle'],
                                      num_workers=settings['num_workers'],
                                      pin_memory=True)

            test_loader = DataLoader(test_data,
                                     batch_size=settings['batch_size'],
                                     num_workers=settings['num_workers'],
                                     pin_memory=True)

    else:
        coco_annotations = COCO(settings['annotation_file'])

        train_data = CocoStuffDataset(dataset_path=path,
                                      filenames=training_data,
                                      coco_annotations=coco_annotations,
                                      mask_type=settings['mask_type'],
                                      transforms=settings['transforms'])
        test_data = CocoStuffDataset(dataset_path=path,
                                     filenames=testing_data,
                                     coco_annotations=coco_annotations,
                                     mask_type=settings['mask_type'],
                                     transforms=settings['transforms'])

        if subset_data:
            subset_train_size = 500
            indices_train = list(range(len(train_data)))
            subset_train_indices = indices_train[:subset_train_size]

            train_sampler = SubsetRandomSampler(subset_train_indices)
            train_loader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=settings['batch_size'],
                                      num_workers=settings['num_workers'],
                                      pin_memory=True)

            subset_test_size = 200
            indices_test = list(range(len(test_data)))
            subset_test_indices = indices_test[:subset_test_size]

            test_sampler = SubsetRandomSampler(subset_test_indices)
            test_loader = DataLoader(test_data,
                                     sampler=test_sampler,
                                     batch_size=settings['batch_size'],
                                     num_workers=settings['num_workers'],
                                     pin_memory=True)

        else:
            # Create the dataloaders
            train_loader = DataLoader(train_data,
                                      batch_size=settings['batch_size'],
                                      shuffle=settings['shuffle'],
                                      num_workers=settings['num_workers'],
                                      pin_memory=True)

            test_loader = DataLoader(test_data,
                                     batch_size=settings['batch_size'],
                                     num_workers=settings['num_workers'],
                                     pin_memory=True)

    return train_loader, test_loader


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def visualize_batch(images: torch.Tensor, masks: torch.Tensor, output_path: str,
                    epoch: int, num_images: int = 3) -> None:
    batch_size = images.size(0)
    # Randomly get 'num_images'
    indices = torch.randperm(batch_size)[:num_images]

    selected_imgs = images[indices]
    selected_masks = masks[indices]

    combined_images: List[torch.Tensor] = []

    for i in range(num_images):
        img = selected_imgs[i]
        mask = selected_masks[i]

        combined_images.append(img)

        img_with_mask = img.clone()
        img_with_mask[:, mask[0] > 0.5] = 1  # Supersede a white mask on top of the image
        combined_images.append(img_with_mask)

    # Create a Tensor from the list
    combined_images = torch.stack(combined_images)

    grid = make_grid(combined_images, nrow=2)
    save_image(grid, os.path.join(output_path, 'visualization_epoch_{}.png'.format(epoch)))


def visualize_multiclass_batch(images: torch.Tensor, masks: torch.Tensor, output_path: str,
                               image_paths: List[str], epoch: int, num_images: int = 3,
                               palette_path='/home/rob/Documents/3_projects/bench/isaid/isaid_mask_palette.json') -> None:
    with open(palette_path, 'r') as f:
        color_map = json.load(f)
        color_map = {int(k): v for k, v in color_map.items()}  # keys must be converted to int

    selected_imgs = images[:num_images]
    selected_masks = masks[:num_images]
    selected_paths = image_paths[:num_images]

    combined_images: List[torch.Tensor] = []

    for i in range(num_images):
        img = selected_imgs[i]
        mask = selected_masks[i].squeeze(0)  # Remove C dim for the mask
        img_path = selected_paths[i]

        # Convert to color image
        mask_rgb = torch.zeros((3, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        for class_idx, color in color_map.items():
            mask_rgb[:, mask == class_idx] = torch.tensor(color, dtype=torch.uint8).unsqueeze(1)

        # Denormalize the image and convert to uint8 for OpenCV compatibility
        img_denorm = (img.clone() * 255).byte().permute(1, 2, 0).numpy()

        # Add the image path text on the image
        fold_nb = img_path.split('/')[-2]
        img_nb = img_path.split('/')[-1].split('.png')[0]
        path_name = fold_nb + '/' + img_nb
        print(f'image path name: {path_name}')
        img_with_text = cv2.putText(img_denorm.copy(), path_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 0, 0), 1)

        # Convert img_with_text back to tensor and append to combined_images
        combined_images.append(torch.tensor(img_with_text).permute(2, 0, 1).float() / 255)

        img_with_mask = torch.tensor(img_with_text).permute(2, 0, 1).float() / 255
        img_with_mask += mask_rgb.float() / 255.0
        combined_images.append(img_with_mask)

    combined_images = torch.stack(combined_images)

    grid = make_grid(combined_images, nrow=2)
    save_image(grid, os.path.join(output_path, 'visualization_multiclass_epoch_{}.png'.format(epoch)))


def generate_color_palette(num_classes: int):
    random.seed(0)  # Fixing the seed for reproducibility
    color_palette = {}
    for i in range(num_classes):
        color_palette[i] = [random.randint(0, 255) for _ in range(3)]
    return color_palette

def visualize_multiclass_batch_with_generated_palette(images: torch.Tensor, masks: torch.Tensor, output_path: str,
                                                      image_paths: List[str], epoch: int, num_images: int = 3) -> None:
    unique_classes = torch.unique(masks)
    color_map = generate_color_palette(len(unique_classes))
    color_map = {int(k): v for k, v in zip(unique_classes.tolist(), color_map.values())}

    selected_imgs = images[:num_images]
    selected_masks = masks[:num_images]
    selected_paths = image_paths[:num_images]

    combined_images: List[torch.Tensor] = []

    for i in range(num_images):
        img = selected_imgs[i]
        mask = selected_masks[i].squeeze(0)  # Remove C dim for the mask
        img_path = selected_paths[i]

        # Convert to color image
        mask_rgb = torch.zeros((3, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        for class_idx, color in color_map.items():
            mask_rgb[:, mask == class_idx] = torch.tensor(color, dtype=torch.uint8).unsqueeze(1)

        # Denormalize the image and convert to uint8 for OpenCV compatibility
        img_denorm = (img.clone() * 255).byte().permute(1, 2, 0).numpy()

        # Add the image path text on the image
        fold_nb = img_path.split('/')[-2]
        img_nb = img_path.split('/')[-1].split('.png')[0]
        path_name = fold_nb + '/' + img_nb
        print(f'image path name: {path_name}')
        img_with_text = cv2.putText(img_denorm.copy(), path_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 0, 0), 1)

        # Convert img_with_text back to tensor and append to combined_images
        combined_images.append(torch.tensor(img_with_text).permute(2, 0, 1).float() / 255)

        img_with_mask = torch.tensor(img_with_text).permute(2, 0, 1).float() / 255
        img_with_mask += mask_rgb.float() / 255.0
        combined_images.append(img_with_mask)

    combined_images = torch.stack(combined_images)

    grid = make_grid(combined_images, nrow=2)
    save_image(grid, os.path.join(output_path, 'visualization_multiclass_epoch_{}.png'.format(epoch)))


if __name__ == "__main__":
    settings_kvasir = {
        "name": "kvasir",
        'path': "/home/rob/Documents/3_projects/bench/kvasir",
        "annotation_file": None,
        'multiclass_palette_path': None,
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

    settings_isaid = {
        'name': "isaid",
        'path': "/home/rob/Documents/3_projects/bench/isaid",
        'mask_type': 'multiclass_semantic',
        "annotation_file": None,
        'multiclass_palette_path': '/home/rob/Documents/3_projects/bench/isaid/isaid_mask_palette.json',
        'transforms': {
            'Resize': None,
            'RandomCrop': None,
            'flip': False,
            'rot': False,
            'Normalize': None
        },
        'batch_size': 3,
        'shuffle': True,
        'num_workers': 1
    }

    settings_coco = {
        'name': "coco",
        'path': "/home/rob/Documents/3_projects/bench/coco/output",
        "annotation_file": "/home/rob/Documents/3_projects/bench/coco/output/tmp_data/stuff_annotations_trainval2017/annotations/stuff_train2017.json",
        'mask_type': 'multiclass_semantic',
        'multiclass_palette_path': None,
        'transforms': {
            'Resize': (256, 256),
            'RandomCrop': None,
            'flip': False,
            'rot': False,
            'Normalize': None
        },
        'batch_size': 3,
        'shuffle': True,
        'num_workers': 1
    }

    fold = 0
    output_path = os.path.join(os.getcwd(), 'outputs')
    epoch = 1

    # dataset choice
    settings = settings_coco
    train_loader, test_loader = get_data_loader(settings, fold, subset_data=True,
                                                annotation_file=settings['annotation_file'])

    for data in train_loader:
        if settings['mask_type'] == 'multiclass_semantic':
            if settings['name'] == 'coco':
                visualize_multiclass_batch_with_generated_palette(data['img'], data['mask'],
                                                                  output_path, data['img_path'],
                                                                  epoch, num_images=3)
            else:
                visualize_multiclass_batch(data['img'], data['mask'],
                                           output_path, data['img_path'],
                                           epoch, num_images=3,
                                           palette_path=settings['multiclass_palette_path'])
        elif settings['mask_type'] == 'single_class':
            visualize_batch(data['img'], data['mask'], output_path, epoch, num_images=3)
        else:
            raise NotImplementedError
        break
