from audioop import avg
from concurrent.futures import process
from dis import dis
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import torch.distributed as dist

import utils

def main(args):
    utils.init_distributed_mode(args)
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        #A: es carrega la dataset de training nomes
        if phase == 'train' and opt['phase'] != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase, num_tasks, global_rank)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase, num_tasks, global_rank)
    logger.info('Initial Dataset Finished')
    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    
    # validation
    avg_psnr = 0.0
    idx = 0

    #A: fa que dintre de la carpeta results es crei una subcarbeta amb nom current_step i aquesta carpeta es la que contindra les carpetes hr lr sr
    result_hr_path = opt['path']['results_hr'].rsplit('/', 1)[0] + '/{}/'.format(current_step) + opt['path']['results_hr'].rsplit('/', 1)[1]
    result_sr_path = opt['path']['results_sr'].rsplit('/', 1)[0] + '/{}/'.format(current_step) + opt['path']['results_sr'].rsplit('/', 1)[1]
    result_lr_path = opt['path']['results_lr'].rsplit('/', 1)[0] + '/{}/'.format(current_step) + opt['path']['results_lr'].rsplit('/', 1)[1]


    os.makedirs('{}'.format(result_hr_path), exist_ok=True)
    os.makedirs('{}'.format(result_sr_path), exist_ok=True)
    os.makedirs('{}'.format(result_lr_path), exist_ok=True)

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    for _,  val_data in enumerate(val_loader):
        idx += 1
        print('index', idx)
        diffusion.feed_data(val_data)
        
        #A: Al fer el test es fa inference sobre el model GaussianDiffusion i s'obte la imatge SR
        diffusion.test(continous=False) #AQUI PETA
        
        visuals = diffusion.get_current_visuals()
        
        #A: HR i LR son les imatges que aportem nosaltres per entrenar i SR es la que es genera al fer inferencia del GaussianDiffusion
        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8 
        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8


        # generation
        Metrics.save_img(
            hr_img, '{}/{}_{}_hr.png'.format(result_hr_path, idx, dist.get_rank()))
        Metrics.save_img(
            sr_img, '{}/{}_{}_sr.png'.format(result_sr_path, idx, dist.get_rank()))
        Metrics.save_img(
            lr_img, '{}/{}_{}_lr.png'.format(result_lr_path, idx, dist.get_rank()))

        tb_logger.add_image(
            'Iter_{}'.format(current_step),
            np.transpose(np.concatenate(
                (sr_img, hr_img), axis=1), [2, 0, 1]),
            idx)
        avg_psnr += Metrics.calculate_psnr(
            sr_img, hr_img)

        if wandb_logger:
            wandb_logger.log_image(
                f'validation_{idx}', 
                np.concatenate((sr_img, hr_img), axis=1)
            )
    
    avg_psnr = torch.Tensor([avg_psnr]).to(dist.get_rank())
    dist.reduce(avg_psnr, 0)
    dist.barrier()

    avg_psnr = avg_psnr.item() / (idx * dist.get_world_size())
    # avg_psnr = avg_psnr.item() / idx
    if avg_psnr >= max_psnr and dist.get_rank() == 0:
        max_psnr = avg_psnr
        diffusion.save_network(current_epoch, current_step, best='psnr_{}'.format(max_psnr))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['train'], schedule_phase='train')
    # log
    if dist.get_rank() == 0:

        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
            current_epoch, current_step, avg_psnr))
        # tensorboard logger
        tb_logger.add_scalar('psnr', avg_psnr, current_step)

    if wandb_logger:
        wandb_logger.log_metrics({
            'validation/val_psnr': avg_psnr,
            'validation/val_step': val_step
        })
        val_step += 1

        if wandb_logger:
            wandb_logger.log_metrics({'epoch': current_epoch-1})



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ffhq_liifsr3_scaler_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    # parser.add_argument('--local_rank', type=int,help='local rank for dist')
    parser.add_argument('-r', '--resume', type=str, default='experiments/sr_ffhq/checkpoint/latest')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    main(args)
