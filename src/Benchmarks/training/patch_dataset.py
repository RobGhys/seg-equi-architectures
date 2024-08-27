from typing import Union

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
import os
import random


class PatchDataset(Dataset):
    def __init__(self, dataset_path, train_mode: bool = True,
                 npz_file: str = None,
                 crop_size: tuple = None,
                 stride: int = None, threshold: float = None,
                 mask_type='single_class',
                 transforms=None, filenames=[]):
        self.dataset_path = dataset_path
        self.filenames = sorted(filenames)
        self.train_mode = train_mode

        if self.train_mode:
            self.data = np.load(npz_file, allow_pickle=True)
            self.crop_size = crop_size
            self.stride = stride
            self.threshold = threshold
            self.ratios = self.data['ratios']
            self._prefilter_images()


        self.mask_type = mask_type
        self.transforms = transforms if transforms is not None else {}

        self.toggle = 0
        self.trigger_mask_step = 2 # Force to get a mask with r > t every 2 steps

        if isinstance(self.transforms['Resize'], list):
            self.transforms['Resize'] = tuple(self.transforms['Resize'])


    def __len__(self):
        return len(self.filenames)

    def _prefilter_images(self):
        valid_filenames = []
        valid_ratios = []

        for filename, img_ratios in zip(self.filenames, self.ratios):
            if any(r > self.threshold for r in img_ratios):
                valid_filenames.append(filename)
                valid_ratios.append(img_ratios)

        self.filenames = valid_filenames
        self.ratios = valid_ratios

    def __getitem__(self, index):
        img_path = os.path.join(self.dataset_path, 'imgs', self.filenames[index])
        mask_path = os.path.join(self.dataset_path, 'masks', self.filenames[index])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.train_mode:
            if self.transforms['Resize']:
                img = F.resize(img, self.transforms['Resize'])
                mask = F.resize(mask, self.transforms['Resize'], interpolation=Image.NEAREST)

            ratios = self.ratios[index]

            # Once, get a mask with r > t, once get a mask with r <= t
            if self.toggle % self.trigger_mask_step == 0:
                valid_indices = [i for i, r in enumerate(ratios) if r > self.threshold]
            else:
                valid_indices = [i for i, r in enumerate(ratios) if r <= self.threshold]

            if len(valid_indices) == 0:
                valid_indices = range(len(ratios))

            selected_index = random.choice(valid_indices)

            num_patches_per_row = (mask.size[0] - self.crop_size[1]) // self.stride + 1
            y = (selected_index // num_patches_per_row) * self.stride
            x = (selected_index % num_patches_per_row) * self.stride

            patch_mask = mask.crop((x, y, x + self.crop_size[1], y + self.crop_size[0]))
            patch_img = img.crop((x, y, x + self.crop_size[1], y + self.crop_size[0]))
        else:
            patch_img = img
            patch_mask = mask

        if self.transforms['flip']:
            if np.random.rand() > 0.5:
                patch_img: Union[Image.Image, torch.Tensor] = F.hflip(patch_img)
                patch_mask: Union[Image.Image, torch.Tensor] = F.hflip(patch_mask)
            if np.random.rand() > 0.5:
                patch_img: Union[Image.Image, torch.Tensor] = F.vflip(patch_img)
                patch_mask: Union[Image.Image, torch.Tensor] = F.vflip(patch_mask)

        if self.transforms['rot']:
            d = np.random.choice([0, 90, 180, 270])
            if d != 0:
                patch_img: Union[Image.Image, torch.Tensor] = F.rotate(patch_img, int(d))
                patch_mask: Union[Image.Image, torch.Tensor] = F.rotate(patch_mask, int(d))

        if not self.train_mode:
            if self.transforms['Resize']:
                patch_img: Union[Image.Image, torch.Tensor] = F.resize(patch_img, self.transforms['Resize'])
                patch_mask: Union[Image.Image, torch.Tensor] = F.resize(
                    patch_mask, self.transforms['Resize'],
                    interpolation=Image.NEAREST)

        patch_img: torch.Tensor = transforms.ToTensor()(patch_img)
        if self.transforms['Normalize']:
            patch_img = transforms.Normalize(*self.transforms['Normalize'])(patch_img)

        if self.mask_type == 'single_class':
            patch_mask = patch_mask.convert('L')
            patch_mask: torch.Tensor = transforms.ToTensor()(patch_mask)
            patch_mask: torch.Tensor = (patch_mask > 0).long()

        if self.train_mode:
            self.toggle += 1

        data = {'img': patch_img, 'mask': patch_mask, 'img_path': img_path, 'mask_path': mask_path}


        return data
