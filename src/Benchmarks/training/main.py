"""
This code train a U-Net model on different preprocessed datasets
accordingly to the _settings.json file (specificities for each dataset)
"""

import argparse
import json
from datetime import datetime

from torch import optim
from torchmetrics.classification import JaccardIndex, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAccuracy

from tqdm import tqdm

from engine import run_epoch_binary_seg, run_epoch_multiclass_seg
from load_data import *
from load_model import *
from scores import DiceLoss, DiceCoeff, IoUCoeff, Precision, Recall, Accuracy, DiceLossMulticlass
from utils import model_sanity_check, launch_weights_and_biases

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Train a U-Net model on different preprocessed datasets')
parser.add_argument('dataset_name', type=str, help='id of the dataset to use', choices=['NucleiSeg', 'kvasir', 'URDE', 'isaid', 'coco'])
parser.add_argument('model_name', type=str, help='model to use',
                    choices=['UNet_vanilla', 'UNet_bcnn', 'UNet_e2cnn'])
parser.add_argument('fold', type=int, help='fold to use', choices=[0, 1, 2, 3, 4])
parser.add_argument('--use_amp', default=False, help='use automatic mixed precision', action='store_true')
parser.add_argument('--save_logs', default=False, help='save the logs of the training', action='store_true')
parser.add_argument('--save_model', default=False, help='save the model', action='store_true')
parser.add_argument('--save_images', default=False,
                    help='save the images with their true and predicted mask while training', action='store_true')
parser.add_argument('--new_model_name', type=str, help='Optional name of the folder to save the results', default=None)
parser.add_argument('--location_lucia', default=False, help='Data are located on lucia', action='store_true')
parser.add_argument('--wandb_api_key', type=str, help='Personal API key for weight and biases logs')
parser.add_argument('--subset_data', default=False, help='Uses a subset of the Dataset', action='store_true')
parser.add_argument('--mac', default=False, help='Uses MAC', action='store_true')

dataset_name = parser.parse_args().dataset_name
model_name = parser.parse_args().model_name
fold = parser.parse_args().fold
use_amp = parser.parse_args().use_amp
save_logs = parser.parse_args().save_logs
save_model = parser.parse_args().save_model
save_images = parser.parse_args().save_images
new_model_name = parser.parse_args().new_model_name
data_location_lucia = parser.parse_args().location_lucia
wandb_api_key = parser.parse_args().wandb_api_key
subset_data = parser.parse_args().subset_data
mac = parser.parse_args().mac

if new_model_name:
    output_path = os.path.join(os.getcwd(), 'outputs', dataset_name, new_model_name, 'fold_' + str(fold))
else:
    output_path = os.path.join(os.getcwd(), 'outputs', dataset_name, model_name, 'fold_' + str(fold))

os.makedirs(output_path, exist_ok=True)

base_dir = os.path.dirname(os.path.abspath(__file__))
if data_location_lucia:
    settings_json = os.path.join(base_dir, '_settings_data.json')
else:
    settings_json = os.path.join(base_dir, '_settings_data_local.json')

# Get the data params
with open(settings_json, 'r') as jsonfile:
    settings = json.load(jsonfile)[dataset_name]

# Load the data
train_loader, test_loader = get_data_loader(settings, fold, subset_data, mac=mac)

# Load the model
model, n_params = getModel(model_name, settings)
model = model.to(device=device)

model_sanity_check(settings, n_params, device, model)
log_wandb = bool(wandb_api_key)
if wandb_api_key:
    launch_weights_and_biases(model=model_name, dataset=dataset_name,
                              settings=settings, fold_nb=fold, api_key=wandb_api_key)
    save_logs = False

# Define the optimizer, the loss and the learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=settings['models']['lr'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.33)
grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

eval_metrics: dict = {}

if model.n_classes > 1:
    eval_metrics['loss_ce'] = nn.CrossEntropyLoss().to(device)
    eval_metrics['dice_criterion'] = DiceLossMulticlass().to(device)
    eval_metrics['IoU_score'] = JaccardIndex(task='multiclass', num_classes=settings['n_classes']).to(device)
    #eval_metrics['f1_score'] = MulticlassF1Score(num_classes=settings['n_classes']).to(device)
    eval_metrics['recall_metric'] = MulticlassRecall(num_classes=settings['n_classes']).to(device)
    eval_metrics['precision_metric'] = MulticlassPrecision(num_classes=settings['n_classes']).to(device)
    eval_metrics['accuracy_metric'] = MulticlassAccuracy(num_classes=settings['n_classes']).to(device)
else:
    eval_metrics['loss_ce'] = nn.BCELoss(reduction='mean').to(device)
    eval_metrics['dice_criterion'] = DiceLoss().to(device)
    eval_metrics['dice_coeff'] = DiceCoeff().to(device)
    eval_metrics['IoU_coeff'] = IoUCoeff().to(device)
    eval_metrics['precision_metric'] = Precision().to(device)
    eval_metrics['recall_metric'] = Recall().to(device)
    eval_metrics['accuracy_metric'] = Accuracy().to(device)

if save_logs:  # Use Tensorboard only if no wandb_api_key is provided
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
        'loss_ce': [], 'loss_dice': [], 'dice_score': [], 'IoU_score': [], 'precision_metric': [], 'recall_metric': [],
        'accuracy_metric': [], 'time': []
    },
    'test': {
        'loss_ce': [], 'loss_dice': [], 'dice_score': [], 'IoU_score': [], 'precision_metric': [], 'recall_metric': [],
        'accuracy_metric': [], 'time': []
    }
}

for epoch in tqdm(range(settings['models']['num_epochs'])):
    combined_loss = False

    if model.n_classes == 1:

        train_results = run_epoch_binary_seg(model, train_loader, optimizer, device, settings,
                                             grad_scaler, use_amp, phase='train', writer=writer, log_wandb=log_wandb,
                                             epoch=epoch, save_images=save_images,
                                             output_path=output_path, eval_metrics=eval_metrics, summary=summary,
                                             combined_loss=combined_loss)

        eval_results = run_epoch_binary_seg(model, test_loader, optimizer, device, settings,
                                            grad_scaler, use_amp, phase='test', writer=writer, log_wandb=log_wandb,
                                            epoch=epoch, save_images=save_images,
                                            output_path=output_path, eval_metrics=eval_metrics, summary=summary,
                                            combined_loss=combined_loss)

        print(f'\nEpoch : {epoch} | dice : {eval_results["dice_score"]:.2f} | IoU : {eval_results["IoU_score"]:.2f} |'
              f'Accuracy : {eval_results["accuracy_metric"]:.2f} | Precision : {eval_results["precision_metric"]:.2f}'
              f'| Recall : {eval_results["recall_metric"]:.2f} | LR : {optimizer.param_groups[0]["lr"]:.5f}')

        if combined_loss:
            scheduler.step(train_results['loss_ce'] + train_results['loss_dice'])
        else:
            scheduler.step(train_results['loss_dice'])
    else:
        if save_images and settings['multiclass_palette_path'] is not None:
            palette_path = settings['multiclass_palette_path']
            with open(palette_path, 'r') as f:
                color_map = json.load(f)
                color_map = {int(k): v for k, v in color_map.items()}
        else:
            color_map = None
        train_results = run_epoch_multiclass_seg(model, train_loader, optimizer, device, settings,
                                                 grad_scaler, use_amp, phase='train', writer=writer, log_wandb=log_wandb,
                                                 epoch=epoch, save_images=save_images,
                                                 output_path=output_path, eval_metrics=eval_metrics, summary=summary,
                                                 combined_loss=combined_loss, color_map=color_map,
                                                 dataset=dataset_name)

        eval_results = run_epoch_multiclass_seg(model, test_loader, optimizer, device, settings,
                                                grad_scaler, use_amp, phase='test', writer=writer, log_wandb=log_wandb,
                                                epoch=epoch, save_images=save_images,
                                                output_path=output_path, eval_metrics=eval_metrics, summary=summary,
                                                combined_loss=combined_loss, color_map=color_map,
                                                dataset=dataset_name)

        print(f'\nEpoch : {epoch} | IoU : {eval_results["IoU_score"]:.2f} |'
              f'Accuracy : {eval_results["accuracy_metric"]:.2f} | Precision : {eval_results["precision_metric"]:.2f}'
              f'| Recall : {eval_results["recall_metric"]:.2f} | LR : {optimizer.param_groups[0]["lr"]:.5f}')

        if combined_loss:
            scheduler.step(train_results['loss_ce'] + train_results['loss_dice'])
        else:
            scheduler.step(train_results['loss_dice'])
    if save_model:
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(output_path, 'checkpoint_epoch_{}.pth'.format(epoch)))

# Save the summary
with open(os.path.join(output_path, 'summary.json'), 'w') as jsonfile:
    json.dump(summary, jsonfile, indent=4)

# Save the settings
with open(os.path.join(output_path, 'settings_used.json'), 'w') as jsonfile:
    json.dump(settings, jsonfile, indent=4)
