import os
import json
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as maskUtils
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class CocoStuffDataset(Dataset):
    def __init__(self, dataset_path, coco_annotations, filenames=[], mask_type='single_class',
                 transforms={'Resize': None, 'RandomCrop': None, 'flip': True, 'rot': True,
                             'Normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))}):
        self.dataset_path = dataset_path
        self.coco = coco_annotations
        self.filenames = sorted(filenames)
        self.mask_type = mask_type
        self.transforms = transforms

        unique_category_ids = sorted(set(cat['id'] for cat in self.coco.dataset['categories']))

        self.class_remap = {cid: idx + 1 for idx, cid in enumerate(unique_category_ids)}  # +1 to leave 0 for background
        self.class_remap[0] = 0  # Assuming 0 is the background

        if isinstance(self.transforms['Resize'], list):
            self.transforms['Resize'] = tuple(self.transforms['Resize'])

        print(f'CocoStuffDataset transforms: {self.transforms}')

    def decode_rle(self, rle, height, width):
        mask = maskUtils.decode(rle)
        return mask.reshape(height, width)

    def __getitem__(self, index, debug=False):
        img_path_relative = self.filenames[index]
        img_path = os.path.join(self.dataset_path, 'imgs', img_path_relative)

        # Extract image id from file name
        image_id = int(os.path.splitext(os.path.basename(img_path_relative))[0])
        img_info = self.coco.imgs[image_id]
        img = Image.open(img_path).convert('RGB')

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            rle = ann['segmentation']
            class_id = ann['category_id']
            decoded_mask = self.decode_rle(rle, img_info['height'], img_info['width'])
            decoded_mask_np = np.array(decoded_mask)
            remapped_class_id = self.class_remap.get(class_id, 0)  # Map class_id to remapped_class_id
            mask[decoded_mask_np > 0] = remapped_class_id

        mask = Image.fromarray(mask)

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
            mask = F.resize(mask, self.transforms['Resize'], interpolation=Image.NEAREST)
            if debug:
                print(f"Resize: img size {img.size}, mask size {mask.size}")
                print(f'type of img: {type(img)} | type of mask: {type(mask)}')

        img = transforms.ToTensor()(img)
        if self.transforms['Normalize']:
            img = transforms.Normalize(*self.transforms['Normalize'])(img)

        if self.mask_type == 'multiclass_semantic':
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        elif self.mask_type == 'single_class':
            mask = transforms.ToTensor()(mask).long()
            mask = (mask > 0).long()

        if debug:
            non_zero_elements = mask[mask != 0]
            print(f'Count of Non-zero elements in mask: {len(non_zero_elements)}')
            print(
                f"Final tensor types and shapes: img -> {type(img)}, shape {img.shape}, mask -> {type(mask)}, shape {mask.shape}")
            print(f'example mask: {mask}')

        data = {
            'img':img,
            'mask': mask,
            'img_path': img_path,
            'mask_path': ''
        }

        return data

    def __len__(self):
        return len(self.filenames)