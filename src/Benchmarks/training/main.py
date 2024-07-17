"""
This code train a U-Net model on different preprocessed datasets
accordingly to the _settings.json file (specificities for each dataset)
"""

import random
import argparse
import json
import os
from datetime import datetime

from torch import optim
from tqdm import tqdm

from load_data import *
from load_model import *
from scores import DiceLoss, DiceCoeff, IoUCoeff, Precision, Recall, Accuracy

from torchmetrics import JaccardIndex

from utils import model_sanity_check
from engine import run_epoch

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
    output_path = os.path.join(os.getcwd(), 'outputs', dataset_name, new_model_name, 'fold_' + str(fold))
else:
    output_path = os.path.join(os.getcwd(), 'outputs', dataset_name, model_name, 'fold_' + str(fold))

os.makedirs(output_path, exist_ok=True)
data_location_lucia = False

if data_location_lucia:
    settings_json = '_settings_data.json'
else:
    settings_json = '_settings_data_local_basic.json'
# Get the data params
with open(settings_json, 'r') as jsonfile:
    settings = json.load(jsonfile)[dataset_name]

# Load the data
train_loader, test_loader = getDataLoader(settings, fold)

# Load the model
model, n_params = getModel(model_name, settings)
model = model.to(device=device)

model_sanity_check(settings, n_params, device, model)

# Define the optimizer, the loss and the learning rate scheduler
# optimizer = optim.RMSprop(model.parameters(), lr=settings['models']['lr'],
#                           weight_decay=1e-6, momentum=0.5, foreach=True)
optimizer = optim.AdamW(model.parameters(), lr=settings['models']['lr'])

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
    precision_metric = Precision()
    recall_metric = Recall()
    accuracy_metric = Accuracy()

if save_logs:
    from torch.utils.tensorboard import SummaryWriter

    # Initialize logging
    now = datetime.now()
    writer = SummaryWriter(os.path.join(output_path,
                                        now.strftime('%m_%d_%Y') + '_' + now.strftime('%H_%M_%S')))
else:
    writer = None

# Train the model
summary = {
    'n_params': n_params,
    'train': {
        'loss_ce': [], 'loss_dice': [], 'dice_score': [], 'IoU_score': [], 'precision': [], 'recall': [],
        'accuracy': [], 'time': []
    },
    'test': {
        'loss_ce': [], 'loss_dice': [], 'dice_score': [], 'IoU_score': [], 'precision': [], 'recall': [], 'accuracy': []
    }
}

for epoch in tqdm(range(settings['models']['num_epochs'])):
    train_results = run_epoch(model, train_loader, optimizer, device, settings,
                              grad_scaler, use_amp, phase='train', writer=writer, epoch=epoch, save_images=save_images,
                              output_path=output_path, BCE_criterion=BCE_criterion, dice_criterion=dice_criterion,
                              dice_coeff=dice_coeff, IoU_coeff=IoU_coeff, precision_metric=precision_metric,
                              recall_metric=recall_metric, accuracy_metric=accuracy_metric)
    summary['train']['loss_ce'].append(train_results['loss_ce'])
    summary['train']['loss_dice'].append(train_results['loss_dice'])
    summary['train']['dice_score'].append(train_results['dice_score'])
    summary['train']['IoU_score'].append(train_results['IoU_score'])
    summary['train']['precision'].append(train_results['precision'])
    summary['train']['recall'].append(train_results['recall'])
    summary['train']['accuracy'].append(train_results['accuracy'])
    summary['train']['time'].append(train_results['time'])

    eval_results = run_epoch(model, test_loader, optimizer, device, settings,
                             grad_scaler, use_amp, phase='test', writer=writer, epoch=epoch, save_images=save_images,
                             output_path=output_path, BCE_criterion=BCE_criterion, dice_criterion=dice_criterion,
                             dice_coeff=dice_coeff, IoU_coeff=IoU_coeff, precision_metric=precision_metric,
                             recall_metric=recall_metric, accuracy_metric=accuracy_metric)
    summary['test']['loss_ce'].append(eval_results['loss_ce'])
    summary['test']['loss_dice'].append(eval_results['loss_dice'])
    summary['test']['dice_score'].append(eval_results['dice_score'])
    summary['test']['IoU_score'].append(eval_results['IoU_score'])
    summary['test']['precision'].append(eval_results['precision'])
    summary['test']['recall'].append(eval_results['recall'])
    summary['test']['accuracy'].append(eval_results['accuracy'])

    print(f'\nEpoch : {epoch} | dice : {eval_results["dice_score"]:.2f} | IoU : {eval_results["IoU_score"]:.2f} |'
          f'Accuracy : {eval_results["accuracy"]:.2f} | Precision : {eval_results["precision"]:.2f}'
          f'| Recall : {eval_results["recall"]:.2f} | LR : {optimizer.param_groups[0]["lr"]:.5f}')

    scheduler.step(train_results['loss_ce'] + train_results['loss_dice'])

    if save_model:
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(output_path, 'checkpoint_epoch_{}.pth'.format(epoch)))

# Save the summary
with open(os.path.join(output_path, 'summary.json'), 'w') as jsonfile:
    json.dump(summary, jsonfile, indent=4)

# Save the settings
with open(os.path.join(output_path, 'settings_used.json'), 'w') as jsonfile:
    json.dump(settings, jsonfile, indent=4)
