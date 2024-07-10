"""
This code train a U-Net model on different preprocessed datasets
accordingly to the _settings.json file (specificities for each dataset)
"""

import argparse
import json
import os
from datetime import datetime
from time import time

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from load_data import *
from load_model import *
from scores import DiceLoss, DiceCoeff, IoUCoeff

from torchvision.utils import save_image
from torchmetrics import JaccardIndex

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Train a U-Net model on different preprocessed datasets')
parser.add_argument('dataset_name', type=str, help='id of the dataset to use', choices=['NucleiSeg', 'kvasir'])
parser.add_argument('model_name', type=str, help='model to use', 
                    choices=['UNet_vanilla', 'UNet_bcnn', 'UNet_e2cnn'])
parser.add_argument('fold', type=int, help='fold to use', choices=[0, 1, 2, 3, 4])
parser.add_argument('--use_amp', default=False, help='use automatic mixed precision', action='store_true')
parser.add_argument('--save_logs', default=False, help='save the logs of the training', action='store_true')
parser.add_argument('--save_model', default=False, help='save the model', action='store_true')
parser.add_argument('--save_images', default=False, 
                    help='save the images for the first batch of each epoch', action='store_true')
parser.add_argument('--new_model_name', type=str, help='Optional name of the folder to save the results', default=None)

dataset_name = parser.parse_args().dataset_name
model_name = parser.parse_args().model_name
fold = parser.parse_args().fold
use_amp = parser.parse_args().use_amp
save_logs = parser.parse_args().save_logs
save_model = parser.parse_args().save_model
save_images = parser.parse_args().save_images
new_model_name = parser.parse_args().new_model_name

if new_model_name:
    output_path = os.path.join(os.getcwd(), 'outputs', dataset_name, new_model_name, 'fold_'+str(fold))
else:
    output_path = os.path.join(os.getcwd(), 'outputs', dataset_name, model_name, 'fold_'+str(fold))

os.makedirs(output_path, exist_ok=True)
data_location_lucia = False

if data_location_lucia:
    settings_json = '_settings_data.json'
else:
    settings_json = '_settings_data_local.json'
# Get the data params
with open(settings_json, 'r') as jsonfile:
    settings = json.load(jsonfile)[dataset_name]

# Load the data
train_loader, test_loader = getDataLoader(settings, fold)

# Load the model
model, n_params = getModel(model_name, settings)
model = model.to(device=device)

print("\n * Number of parameters:", n_params)

# Small test of the model
x = torch.randn((settings['batch_size'], settings['in_channels'], 
                 settings['models']['input_size'], 
                 settings['models']['input_size'])).to(device=device)
print("\n * Sanity check of the model:\n",
      "\tinput shape:", x.shape,
      "\n\toutput shape:", model(x).shape)

# Define the optimizer, the loss and the learning rate scheduler
optimizer = optim.RMSprop(model.parameters(), lr=settings['models']['lr'], 
                          weight_decay=1e-6, momentum=0.5, foreach=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.33)
grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
if model.n_classes > 1:
    criterion = nn.CrossEntropyLoss()
    jaccard = JaccardIndex(task='multilabel', num_classes=settings['n_classes']).to(device)
else:
    BCE_criterion = nn.BCELoss(reduction='mean')
    dice_criterion = DiceLoss()
    dice_coeff = DiceCoeff()
    IoU_coeff = IoUCoeff()

if save_logs:
    from torch.utils.tensorboard import SummaryWriter
    # Initialize logging
    now = datetime.now()
    writer = SummaryWriter(os.path.join(output_path, 
                                        now.strftime('%m_%d_%Y')+'_'+now.strftime('%H_%M_%S')))
else:
    writer = None

# Train the model
summary = {
    'n_params': n_params,
    'train': {
        'loss_ce': [], 'loss_dice': [], 'dice_score': [], 'IoU_score': [], 'time': []
        }, 
    'test': {
        'loss_ce': [], 'loss_dice': [], 'dice_score': [], 'IoU_score': []
        }
    }
for epoch in tqdm(range(settings['models']['num_epochs'])):

    # Training loop
    model.train()
    epoch_loss_ce = 0
    epoch_loss_dice = 0
    epoch_dice_score = 0
    epoch_iou_score = 0
    n = 0
    start_time = time()
    for (imgs, masks) in train_loader:
        imgs, masks = imgs.to(device, dtype=torch.float32), masks.to(device, dtype=torch.float32)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=use_amp):
            masks_pred = F.sigmoid(model(imgs))
            if settings['n_classes'] == 1:
                loss_ce = BCE_criterion(masks_pred, masks)
                loss_dice = dice_criterion(masks_pred, masks)
                dice_score = dice_coeff(masks_pred, masks)
                iou_score = IoU_coeff(masks_pred, masks)
            else:
                raise NotImplementedError("Multiclass dice score not implemented")
            loss = loss_ce + loss_dice

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        # Save the global losses
        epoch_loss_ce += loss_ce.item()
        epoch_loss_dice += loss_dice.item()
        epoch_dice_score += dice_score.item()
        epoch_iou_score += iou_score.item()

    if writer:
        writer.add_scalar('Loss/train_ce', epoch_loss_ce, epoch)
        writer.add_scalar('Loss/train_dice', epoch_loss_dice, epoch)
        writer.add_scalar('Dice/train', epoch_dice_score/len(train_loader), epoch)
        writer.add_scalar('IoU/train', epoch_iou_score/len(train_loader), epoch)
        writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Time', time()-start_time, epoch)

    summary['train']['loss_ce'].append(epoch_loss_ce)
    summary['train']['loss_dice'].append(epoch_loss_dice)
    summary['train']['dice_score'].append(epoch_dice_score/len(train_loader))
    summary['train']['IoU_score'].append(epoch_iou_score/len(train_loader))
    summary['train']['time'].append(time()-start_time)

    # Testing loop
    model.eval()
    epoch_loss_ce = 0
    epoch_loss_dice = 0
    epoch_dice_score = 0
    epoch_iou_score = 0
    for i, (imgs, masks) in enumerate(test_loader):
        imgs, masks = imgs.to(device, dtype=torch.float32), masks.to(device, dtype=torch.float32)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=use_amp):
            masks_pred = F.sigmoid(model(imgs))
            if settings['n_classes'] == 1:
                loss_ce = BCE_criterion(masks_pred, masks)
                loss_dice = dice_criterion(masks_pred, masks)
                dice_score = dice_coeff(masks_pred, masks)
                iou_score = IoU_coeff(masks_pred, masks)
            else:
                raise NotImplementedError("Multiclass dice score not implemented")
            loss = loss_ce + loss_dice

        # Save the global losses
        epoch_loss_ce += loss_ce.item()
        epoch_loss_dice += loss_dice.item()
        epoch_dice_score += dice_score.item()
        epoch_iou_score += iou_score.item()

        # Save the images
        if i == 0 and save_images:
            save_image(imgs, os.path.join(output_path, 'imgs_epoch_{}.png'.format(epoch)))
            save_image(masks_pred, os.path.join(output_path, 'preds_epoch_{}.png'.format(epoch)))
            save_image(masks.float(), os.path.join(output_path, 'masks_epoch_{}.png'.format(epoch)))

    if writer:
        writer.add_scalar('Loss/test_ce', epoch_loss_ce, epoch)
        writer.add_scalar('Loss/test_dice', epoch_loss_dice, epoch)
        writer.add_scalar('Dice/test', epoch_dice_score/len(test_loader), epoch)
        writer.add_scalar('IoU/test', epoch_iou_score/len(test_loader), epoch)

    summary['test']['loss_ce'].append(epoch_loss_ce)
    summary['test']['loss_dice'].append(epoch_loss_dice)
    summary['test']['dice_score'].append(epoch_dice_score/len(test_loader))
    summary['test']['IoU_score'].append(epoch_iou_score/len(test_loader))

    # Update the learning rate
    scheduler.step(epoch_loss_ce+epoch_loss_dice)

    # Save the model
    if save_model:
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(output_path, 'checkpoint_epoch_{}.pth'.format(epoch)))

# Save the summary
with open(os.path.join(output_path, 'summary.json'), 'w') as jsonfile:
    json.dump(summary, jsonfile, indent=4)

# Save the settings
with open(os.path.join(output_path, 'settings_used.json'), 'w') as jsonfile:
    json.dump(settings, jsonfile, indent=4)