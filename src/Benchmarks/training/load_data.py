import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

from PIL import Image
import os



class BasicDataset(Dataset):
    """
    Images are supposed to be RGB
    Masks are supposed to be L (binary grayscale)
    """
    def __init__(self, dataset_path, filenames=[], mask_type='single_class',
                 transforms={'Resize': None, 'RandomCrop': None, 
                             'flip': True, 'rot': True, 
                             'Normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))}):
        self.dataset_path = dataset_path
        self.filenames = filenames
        self.mask_type = mask_type
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset_path, 'imgs', self.filenames[index])).convert('RGB')

        if self.mask_type == 'single_class':
            mask = Image.open(os.path.join(self.dataset_path, 'masks', self.filenames[index])).convert('L')
        elif self.mask_type == 'multiclass_semantic':
            raise NotImplementedError("Not implemented yet")
        elif self.mask_type == 'multiclass_instance':
            raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError("Unknown mask type: should be 'single_class', 'multiclass_semantic' or 'multiclass_instance'")

        tf = [transforms.ToTensor()]
        if self.transforms['Resize']:
            tf.append(transforms.Resize(self.transforms['Resize']))
        if self.transforms['Normalize']:
            tf.append(transforms.Normalize(*self.transforms['Normalize']))
        tf_im = transforms.Compose(
            tf
        )
        tf_ma = transforms.Compose(
            tf[:-1]
        )

        img = tf_im(img)
        mask = tf_ma(mask)

        if self.transforms['RandomCrop']:
            i, j, h, w = transforms.RandomCrop.get_params(img, self.transforms['RandomCrop'])
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
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

        return img, mask

    def __len__(self):
        return len(self.filenames)
    


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
    test_loader = DataLoader(test_data, batch_size=settings['batch_size'],
                             shuffle=settings['shuffle'], num_workers=settings['num_workers'])
    
    return train_loader, test_loader