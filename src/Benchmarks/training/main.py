"""
This code train a U-Net model on different preprocessed datasets
accordingly to the _settings.json file (specificities for each dataset)
"""

import argparse
from datetime import datetime

from torch import optim
from torchmetrics.classification import JaccardIndex, MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, \
    AveragePrecision
from tqdm import tqdm

from engine import run_epoch_binary_seg, run_epoch_multiclass_seg
from load_data import *
from load_model import *
from scores import DiceLoss, DiceCoeff, IoUCoeff, Precision, Recall, Accuracy, DiceLossMulticlass
from utils import model_sanity_check, launch_weights_and_biases, save_summary_and_settings, load_summary_and_settings

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
parser.add_argument('--rob', default=False, help='Uses Rob local data', action='store_true')
parser.add_argument('--freq-save-model', type=int, help='Frequency for weights and summary save points', default=100)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)

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
resume = parser.parse_args().resume
start_epoch = parser.parse_args().start_epoch
rob = parser.parse_args().rob
freq_save_model = parser.parse_args().freq_save_model

if new_model_name:
    output_path = os.path.join(os.getcwd(), 'outputs', dataset_name, new_model_name, 'fold_' + str(fold))
else:
    output_path = os.path.join(os.getcwd(), 'outputs', dataset_name, model_name, 'fold_' + str(fold))

os.makedirs(output_path, exist_ok=True)

base_dir = os.path.dirname(os.path.abspath(__file__))
if data_location_lucia:
    settings_json = os.path.join(base_dir, '_settings_data.json')
elif rob:
    settings_json = os.path.join(base_dir, '_settings_data_local_rob.json')
else:
    settings_json = os.path.join(base_dir, '_settings_data_local.json')

# Get the data params
with open(settings_json, 'r') as jsonfile:
    settings = json.load(jsonfile)[dataset_name]

# Load the data
train_loader, test_loader = get_data_loader(settings, fold, subset_data)

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
    eval_metrics['average_precision'] = AveragePrecision(task='multiclass', num_classes=settings['n_classes']).to(device)
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

if resume:
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device=device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                resume, checkpoint["epoch"]
            )
        )
        print(f"Model device after loading checkpoint: {next(model.parameters()).device}")
        print(f"Optimizer device after loading checkpoint: {device}")
    else:
        print("=> no checkpoint found at '{}'".format(resume))

# Train the model
if resume:
    summary, settings = load_summary_and_settings(output_path, start_epoch=start_epoch)

else:
    summary = {
        'n_params': n_params,
        'train': {
            'loss_ce': [], 'loss_dice': [], 'dice_score': [], 'IoU_score': [], 'average_precision' : [],
            'precision_metric': [], 'recall_metric': [], 'accuracy_metric': [], 'time': []
        },
        'test': {
            'loss_ce': [], 'loss_dice': [], 'dice_score': [], 'average_precision': [], 'IoU_score': [],
            'precision_metric': [], 'recall_metric': [], 'accuracy_metric': [], 'time': []
        }
    }

epoch = start_epoch

for epoch in tqdm(range(start_epoch, settings['models']['num_epochs'])):
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

        print(f'\nEpoch : {epoch + 1 } | dice : {eval_results["dice_score"]:.2f} | IoU : {eval_results["IoU_score"]:.2f} |'
              f'Accuracy : {eval_results["accuracy_metric"]:.2f} | Precision : {eval_results["precision_metric"]:.2f}'
              f'| Recall : {eval_results["recall_metric"]:.2f} | LR : {optimizer.param_groups[0]["lr"]:.5f}')

        if combined_loss:
            scheduler.step(train_results['loss_ce'] + train_results['loss_dice'])
        else:
            scheduler.step(train_results['loss_dice'])

        if (epoch + 1) % freq_save_model == 0 or (epoch + 1) == settings['models']['num_epochs']:
            save_summary_and_settings(summary, settings, output_path, epoch)
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
                                                 dataset=dataset_name, model_name=model_name, freq_save_model=freq_save_model)

        eval_results = run_epoch_multiclass_seg(model, test_loader, optimizer, device, settings,
                                                grad_scaler, use_amp, phase='test', writer=writer, log_wandb=log_wandb,
                                                epoch=epoch, save_images=save_images,
                                                output_path=output_path, eval_metrics=eval_metrics, summary=summary,
                                                combined_loss=combined_loss, color_map=color_map,
                                                dataset=dataset_name, model_name=model_name, freq_save_model=freq_save_model)
        print(f'\nEpoch : {epoch + 1} | IoU : {eval_results["IoU_score"]:.2f} |'
              f'Accuracy : {eval_results["accuracy_metric"]:.2f} | Precision : {eval_results["precision_metric"]:.2f}'
              f'| Recall : {eval_results["recall_metric"]:.2f} | LR : {optimizer.param_groups[0]["lr"]:.5f}')

        if combined_loss:
            scheduler.step(train_results['loss_ce'] + train_results['loss_dice'])
        else:
            scheduler.step(train_results['loss_dice'])
        if (epoch + 1) % freq_save_model == 0 or (epoch + 1) == settings['models']['num_epochs']:
            save_summary_and_settings(summary, settings, output_path, epoch)

    if save_model and (epoch + 1) % freq_save_model == 0:
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(output_path, 'checkpoint_epoch_{}.pth'.format(epoch)))

# save at the end
save_summary_and_settings(summary, settings, output_path, epoch)

