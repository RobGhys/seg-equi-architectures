from src.Benchmarks.training.utils import save_image_output
from time import time
import torch
import torch.nn.functional as F

def run_epoch(model, data_loader, optimizer, device, settings, grad_scaler, use_amp,
              phase='train', writer=None, epoch=0, save_images=False, output_path=None,
              BCE_criterion=None, dice_criterion=None, dice_coeff=None,
              IoU_coeff=None, precision_metric=None, recall_metric=None, accuracy_metric=None):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    epoch_loss_ce = 0
    epoch_loss_dice = 0
    epoch_dice_score = 0
    epoch_iou_score = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_accuracy = 0

    start_time = time()
    for i, (imgs, masks) in enumerate(data_loader):
        imgs, masks = imgs.to(device, dtype=torch.float32), masks.to(device, dtype=torch.float32)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=use_amp):
            masks_pred = F.sigmoid(model(imgs))
            if settings['n_classes'] == 1:
                masks_pred_bin = (masks_pred > 0.5).float()

                loss_ce = BCE_criterion(masks_pred, masks)
                loss_dice = dice_criterion(masks_pred, masks)
                dice_score = dice_coeff(masks_pred, masks)
                iou_score = IoU_coeff(masks_pred, masks)

                precision = precision_metric(masks_pred_bin, masks)
                recall = recall_metric(masks_pred_bin, masks)
                accuracy = accuracy_metric(masks_pred_bin, masks)
            else:
                raise NotImplementedError("Multiclass dice score not implemented")
            loss = loss_ce + loss_dice

        if phase == 'train':
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

        epoch_loss_ce += loss_ce.item()
        epoch_loss_dice += loss_dice.item()
        epoch_dice_score += dice_score.item()
        epoch_iou_score += iou_score.item()
        epoch_precision += precision.item()
        epoch_recall += recall.item()
        epoch_accuracy += accuracy.item()

        if phase == 'test' and i == 0 and (epoch + 1) % 10 == 0 and save_images:
            save_image_output(imgs, masks, masks_pred, output_path, epoch, i)

    avg_epoch_loss_ce = epoch_loss_ce / len(data_loader)
    avg_epoch_loss_dice = epoch_loss_dice / len(data_loader)
    avg_epoch_dice_score = epoch_dice_score / len(data_loader)
    avg_epoch_iou_score = epoch_iou_score / len(data_loader)
    avg_epoch_precision = epoch_precision / len(data_loader)
    avg_epoch_recall = epoch_recall / len(data_loader)
    avg_epoch_accuracy = epoch_accuracy / len(data_loader)

    if writer:
        writer.add_scalar(f'Loss/{phase}_ce', avg_epoch_loss_ce, epoch)
        writer.add_scalar(f'Loss/{phase}_dice', avg_epoch_loss_dice, epoch)
        writer.add_scalar(f'Dice/{phase}', avg_epoch_dice_score, epoch)
        writer.add_scalar(f'IoU/{phase}', avg_epoch_iou_score, epoch)
        writer.add_scalar(f'Precision/{phase}', avg_epoch_precision, epoch)
        writer.add_scalar(f'Recall/{phase}', avg_epoch_recall, epoch)
        writer.add_scalar(f'Accuracy/{phase}', avg_epoch_accuracy, epoch)
        if phase == 'train':
            writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Time', time() - start_time, epoch)

    return {
        'loss_ce': avg_epoch_loss_ce,
        'loss_dice': avg_epoch_loss_dice,
        'dice_score': avg_epoch_dice_score,
        'IoU_score': avg_epoch_iou_score,
        'precision': avg_epoch_precision,
        'recall': avg_epoch_recall,
        'accuracy': avg_epoch_accuracy,
        'time': time() - start_time if phase == 'train' else None
    }
