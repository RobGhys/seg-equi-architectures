import shutil

from src.Benchmarks.training.utils import save_image_output, save_multiclass_image_output, visualize_multiclass_batch_with_generated_palette
from time import time
import torch
import torch.nn.functional as F
import wandb
import os

def run_epoch_binary_seg(model, data_loader, optimizer, device,
                         settings, grad_scaler, use_amp,
                         phase='train', writer=None, log_wandb=False,
                         epoch=0, save_images=False,
                         output_path=None,
                         eval_metrics=None,
                         summary=None, save_img_freq: int = 20,
                         combined_loss=False):
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
    for i, (data) in enumerate(data_loader):
        imgs, masks = data['img'].to(device, dtype=torch.float32), data['mask'].to(device, dtype=torch.float32)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=use_amp):
            masks_pred = F.sigmoid(model(imgs))


            if settings['n_classes'] == 1:
                masks_pred_bin = (masks_pred > 0.5).float()

                loss_ce = eval_metrics['loss_ce'](masks_pred, masks)
                loss_dice = eval_metrics['dice_criterion'](masks_pred, masks)

                dice_score = eval_metrics['dice_coeff'](masks_pred_bin, masks)
                iou_score = eval_metrics['IoU_coeff'](masks_pred_bin, masks)
                precision = eval_metrics['precision_metric'](masks_pred_bin, masks)
                recall = eval_metrics['recall_metric'](masks_pred_bin, masks)
                accuracy = eval_metrics['accuracy_metric'](masks_pred_bin, masks)
            else:  # > 1
                raise NotImplementedError("Method only available for binary segmentation.")
            if combined_loss:
                loss = loss_ce + loss_dice
            else:
                loss = loss_dice

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

        if phase == 'test' and i == 0 and (epoch + 1) % save_img_freq == 0 and save_images:
            save_image_output(imgs, masks, masks_pred, output_path, epoch, i)

    avg_epoch_loss_ce = epoch_loss_ce / len(data_loader)
    avg_epoch_loss_dice = epoch_loss_dice / len(data_loader)
    avg_epoch_dice_score = epoch_dice_score / len(data_loader)
    avg_epoch_iou_score = epoch_iou_score / len(data_loader)
    avg_epoch_precision = epoch_precision / len(data_loader)
    avg_epoch_recall = epoch_recall / len(data_loader)
    avg_epoch_accuracy = epoch_accuracy / len(data_loader)

    if log_wandb:
        log_data = {
            f"CE Loss/{phase}": avg_epoch_loss_ce,
            f"Dice Loss/{phase}": avg_epoch_loss_dice,
            f"Total Loss/{phase}": avg_epoch_loss_ce + avg_epoch_loss_dice,
            f"Dice Score/{phase}": avg_epoch_dice_score,
            f"Accuracy/{phase}": avg_epoch_accuracy,
            f"Precision/{phase}": avg_epoch_precision,
            f"Recall/{phase}": avg_epoch_recall,
            f"IoU/{phase}": avg_epoch_iou_score
        }
        if phase == 'train':
            log_data["Learning Rate"] = optimizer.param_groups[0]['lr']
        wandb.log(log_data, step=epoch)

    elif writer:
        writer.add_scalar(f'Loss/{phase}_ce', avg_epoch_loss_ce, epoch)
        writer.add_scalar(f'Loss/{phase}_dice', avg_epoch_loss_dice, epoch)
        writer.add_scalar(f'Loss/{phase}_total', avg_epoch_loss_ce + avg_epoch_loss_dice, epoch)
        writer.add_scalar(f'Dice/{phase}', avg_epoch_dice_score, epoch)
        writer.add_scalar(f'IoU/{phase}', avg_epoch_iou_score, epoch)
        writer.add_scalar(f'Precision/{phase}', avg_epoch_precision, epoch)
        writer.add_scalar(f'Recall/{phase}', avg_epoch_recall, epoch)
        writer.add_scalar(f'Accuracy/{phase}', avg_epoch_accuracy, epoch)
        if phase == 'train':
            writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Time', time() - start_time, epoch)

    summary[phase]['loss_ce'].append(avg_epoch_loss_ce)
    summary[phase]['loss_dice'].append(avg_epoch_loss_dice)
    summary[phase]['dice_score'].append(avg_epoch_dice_score)
    summary[phase]['IoU_score'].append(avg_epoch_iou_score)
    summary[phase]['precision_metric'].append(avg_epoch_precision)
    summary[phase]['recall_metric'].append(avg_epoch_recall)
    summary[phase]['accuracy_metric'].append(avg_epoch_accuracy)
    summary[phase]['time'].append(time() - start_time)


    return {
        'loss_ce': avg_epoch_loss_ce,
        'loss_dice': avg_epoch_loss_dice,
        'dice_score': avg_epoch_dice_score,
        'IoU_score': avg_epoch_iou_score,
        'precision_metric': avg_epoch_precision,
        'recall_metric': avg_epoch_recall,
        'accuracy_metric': avg_epoch_accuracy,
        'time': time() - start_time
    }

def run_epoch_multiclass_seg(model, data_loader, optimizer, device, settings, grad_scaler, use_amp,
                             phase='train', writer=None, log_wandb=False, epoch=0, save_images=False, output_path=None,
                             eval_metrics=None, summary=None, save_img_freq: int = 1, combined_loss=False,
                             color_map=None, dataset: str = 'default', model_name: str = 'UNet_vanilla', freq_save_model: int = 2):
    if phase == 'train':
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

    epoch_loss_ce = 0
    epoch_loss_dice = 0

    epoch_iou_score = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    epoch_accuracy = 0.0
    epoch_average_precision = 0.0

    start_time = time()

    for i, (data) in enumerate(data_loader):
        imgs, masks = data['img'].to(device, dtype=torch.float32), data['mask'].to(device, dtype=torch.long).squeeze(1)

        if phase != 'train':
            with torch.no_grad():
                with torch.autocast(device.type, enabled=use_amp):
                    masks_pred = model(imgs)
                    if settings['n_classes'] > 1:
                        loss_ce = eval_metrics['loss_ce'](masks_pred, masks)
                        loss_dice = eval_metrics['dice_criterion'](masks_pred, masks)

                        iou_score = eval_metrics['IoU_score'](masks_pred, masks)
                        recall_score = eval_metrics['recall_metric'](masks_pred, masks)
                        precision_score = eval_metrics['precision_metric'](masks_pred, masks)
                        accuracy_score = eval_metrics['accuracy_metric'](masks_pred, masks)
                        average_precision = eval_metrics['average_precision'](masks_pred, masks)
                    else:
                        raise NotImplementedError("Method only available for multilabel segmentation.")
                    loss = loss_ce if combined_loss else loss_ce
        else:
            with torch.autocast(device.type, enabled=use_amp):
                masks_pred = model(imgs)
                if settings['n_classes'] > 1:
                    loss_ce = eval_metrics['loss_ce'](masks_pred, masks)
                    loss_dice = eval_metrics['dice_criterion'](masks_pred, masks)

                    # save memory
                    with torch.no_grad():
                        masks_pred = masks_pred.detach()
                        iou_score = eval_metrics['IoU_score'](masks_pred, masks)
                        recall_score = eval_metrics['recall_metric'](masks_pred, masks)
                        precision_score = eval_metrics['precision_metric'](masks_pred, masks)
                        accuracy_score = eval_metrics['accuracy_metric'](masks_pred, masks)
                        average_precision = eval_metrics['average_precision'](masks_pred, masks)

                        #torch.cuda.empty_cache()

                else:
                    raise NotImplementedError("Method only available for multilabel segmentation.")
                loss = loss_ce if combined_loss else loss_ce
        if phase == 'train':
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

        epoch_loss_ce += loss_ce.item()
        epoch_loss_dice += loss_dice.item()
        epoch_iou_score += iou_score.item()
        epoch_recall += recall_score.item()
        epoch_precision += precision_score.item()
        epoch_accuracy += accuracy_score.item()
        if not torch.isnan(average_precision).any() and not torch.isinf(average_precision).any():
            epoch_average_precision += average_precision.item()
        else:
            print(f"Warning: Average precision contains NaN or Inf values at epoch {epoch}.")
            epoch_average_precision += 0

        if 'average_precision' in eval_metrics:
            metric = eval_metrics['average_precision']
            if hasattr(metric, 'reset'):
                metric.reset()

        torch.cuda.empty_cache()

        # if phase == 'test' and i == 0 and (epoch + 1) % save_img_freq == 0 and save_images:
        #     if dataset == 'coco':
        #         visualize_multiclass_batch_with_generated_palette(data['img'], data['mask'], output_path,
        #                                                           data['img_path'], epoch, num_images=3)
        #     else:
        #         save_multiclass_image_output(imgs, masks, masks_pred, output_path, epoch, i, color_map=color_map)

    avg_epoch_loss_ce = epoch_loss_ce / len(data_loader)
    avg_epoch_loss_dice = epoch_loss_dice / len(data_loader)
    avg_epoch_iou_score = epoch_iou_score / len(data_loader)
    avg_epoch_recall = epoch_recall / len(data_loader)
    avg_epoch_precision = epoch_precision / len(data_loader)
    avg_epoch_accuracy = epoch_accuracy / len(data_loader)
    avg_epoch_average_precision = epoch_average_precision / len(data_loader)

    if log_wandb:
        log_data = {
            f"CE Loss/{phase}": avg_epoch_loss_ce,
            f"Dice Loss/{phase}": avg_epoch_loss_dice,
            f"IoU/{phase}": avg_epoch_iou_score,
            f"Recall/{phase}": avg_epoch_recall,
            f"Precision/{phase}": avg_epoch_precision,
            f"Accuracy/{phase}": avg_epoch_accuracy,
            f"Average_Precision/{phase}": avg_epoch_average_precision
        }
        if phase == 'train':
            log_data["Learning Rate"] = optimizer.param_groups[0]['lr']
        wandb.log(log_data, step=epoch)

    elif writer:
        writer.add_scalar(f'Loss/{phase}_ce', avg_epoch_loss_ce, epoch)
        writer.add_scalar(f'Loss/{phase}_dice', avg_epoch_loss_dice, epoch)
        writer.add_scalar(f'IoU/{phase}', avg_epoch_iou_score, epoch)
        writer.add_scalar(f'Recall/{phase}', avg_epoch_recall, epoch)
        writer.add_scalar(f'Precision/{phase}', avg_epoch_precision, epoch)
        writer.add_scalar(f'Accuracy/{phase}', avg_epoch_accuracy, epoch)
        writer.add_scalar(f'Average_Precision/{phase}', avg_epoch_average_precision, epoch)

        if phase == 'train':
            writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Time', time() - start_time, epoch)

    summary[phase]['loss_ce'].append(avg_epoch_loss_ce)
    summary[phase]['loss_dice'].append(avg_epoch_loss_dice)
    summary[phase]['IoU_score'].append(avg_epoch_iou_score)
    summary[phase]['recall_metric'].append(avg_epoch_recall)
    summary[phase]['precision_metric'].append(avg_epoch_precision)
    summary[phase]['accuracy_metric'].append(avg_epoch_accuracy)
    summary[phase]['average_precision'].append(avg_epoch_average_precision)
    summary[phase]['time'].append(time() - start_time)

    # Save checkpoint
    if (epoch + 1) % freq_save_model == 0 and phase == 'eval':
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch + 1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': model_name,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(output_path, checkpoint_name))
        print(f"Model checkpoint saved at epoch {epoch + 1} to {checkpoint_name}.")

    del eval_metrics
    torch.cuda.empty_cache()

    return {
        'loss_ce': avg_epoch_loss_ce,
        'loss_dice': avg_epoch_loss_dice,
        'IoU_score': avg_epoch_iou_score,
        'recall_metric': avg_epoch_recall,
        'precision_metric': avg_epoch_precision,
        'accuracy_metric': avg_epoch_accuracy,
        'average_precision': avg_epoch_average_precision,
        'time': time() - start_time
    }

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')