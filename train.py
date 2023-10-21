import copy
import math
import os
from functools import partial

import wandb
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import yaml

from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from datasets.pdbbind import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch, loss_function, inference_epoch
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage
from utils.so2 import SO2VESchedule
import pdb

def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir):
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0
    loss_fn = partial(loss_function, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
                      tor_weight=args.tor_weight, chi_weight=args.chi_weight, no_torsion=args.no_torsion, no_chi_angle=args.no_chi_angle)
    print("Starting training...")
    for epoch in range(args.n_epochs):
        if epoch % 5 == 0: print("Run name: ", args.run_name)
        logs = {}
        train_losses = train_epoch(model, train_loader, optimizer, args.device, t_to_sigma, loss_fn, ema_weights)
        print("Epoch {}: Training loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}    chi {:.4f}"
              .format(epoch, train_losses['loss'], train_losses['tr_loss'], train_losses['rot_loss'],
                      train_losses['tor_loss'], train_losses['chi_loss']))

        ema_weights.store(model.parameters())
        if args.use_ema: ema_weights.copy_to(model.parameters()) # load ema parameters into model for running validation and inference
        val_losses = test_epoch(model, val_loader, args.device, t_to_sigma, loss_fn, args.test_sigma_intervals)
        print("Epoch {}: Validation loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}   chi {:.4f}"
              .format(epoch, val_losses['loss'], val_losses['tr_loss'], val_losses['rot_loss'], val_losses['tor_loss'], val_losses['chi_loss']))

        if args.val_inference_freq != None and (epoch + 1) % args.val_inference_freq == 0:
            inf_metrics = inference_epoch(model, val_loader.dataset.complex_graphs[:args.num_inference_complexes], args.device, t_to_sigma, args)
            print("Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f}"
                  .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5']))
            print(f"atom_rmsd_per_residue: {inf_metrics['atom_rmsd_per_residue'].mean():<20}"
                  f"chi_0_mae_deg: {inf_metrics['chi_0_ae_deg'].mean():<20}"
                  f"chi_1_mae_deg: {inf_metrics['chi_1_ae_deg'].mean():<20}"
                  f"chi_2_mae_deg: {inf_metrics['chi_2_ae_deg'].mean():<20}"
                  f"chi_3_mae_deg: {inf_metrics['chi_3_ae_deg'].mean():<20}")
            logs.update({"val_inference/"+ k: v for k, v in inf_metrics.items()}, step=epoch)
            # pdb.set_trace()
        if not args.use_ema: ema_weights.copy_to(model.parameters())
        ema_state_dict = copy.deepcopy(model.module.state_dict() if args.device.type == 'cuda' else model.state_dict())
        ema_weights.restore(model.parameters())

        if args.wandb:
            logs.update({"train/" + k: v for k, v in train_losses.items()})
            logs.update({"validation/" + k: v for k, v in val_losses.items()})
            logs['current_lr'] = optimizer.param_groups[0]['lr']
            # wandb.log(logs, step=epoch + 1)
            wandb.log(logs, step=epoch) # don't log number of parameters

        state_dict = model.module.state_dict() if args.device.type == 'cuda' else model.state_dict()
        if args.inference_earlystop_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
                 args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value):
            best_val_inference_value = logs[args.inference_earlystop_metric]
            best_val_inference_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_inference_epoch_model.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'))
        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))

        if scheduler:
            if args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_losses['loss'])

        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'ema_weights': ema_weights.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    print("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))


def main_function(device):
    args = parse_train_args()
    args.device = device

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert (args.scheduler_patience > args.val_inference_freq) # otherwise we will just stop training after args.scheduler_patience epochs
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # construct loader
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    so2_1pi_periodic = SO2VESchedule(pi_periodic=True, cache_folder=args.diffusion_cache_folder, 
                                     sigma_min=args.chi_sigma_min, sigma_max=args.chi_sigma_max, 
                                     annealed_temp=args.chi_annealed_temp, mode=args.chi_mode)
    so2_2pi_periodic = SO2VESchedule(pi_periodic=False, cache_folder=args.diffusion_cache_folder, 
                                     sigma_min=args.chi_sigma_min, sigma_max=args.chi_sigma_max, 
                                     annealed_temp=args.chi_annealed_temp, mode=args.chi_mode)
    
    so2_periodic = [so2_1pi_periodic, so2_2pi_periodic]
    train_loader, val_loader = construct_loader(args, t_to_sigma, 
                                                so2_periodic)
    # pdb.set_trace()
    model = get_model(args, args.device, t_to_sigma=t_to_sigma, so2_periodic=so2_periodic)
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min')
    ema_weights = ExponentialMovingAverage(model.parameters(),decay=args.ema_rate)

    if args.restart_dir:
        try:
            dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=torch.device('cpu'))
            if args.restart_lr is not None: dict['optimizer']['param_groups'][0]['lr'] = args.restart_lr
            optimizer.load_state_dict(dict['optimizer'])
            model.module.load_state_dict(dict['model'], strict=True)
            if hasattr(args, 'ema_rate'):
                ema_weights.load_state_dict(dict['ema_weights'], device=args.device)
            print("Restarting from epoch", dict['epoch'])
        except Exception as e:
            print("Exception", e)
            dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
            model.module.load_state_dict(dict, strict=True)
            print("Due to exception had to take the best epoch and no optimiser")

    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')

    if args.wandb:
        wandb.init(
            entity='kirito_asuna',
            dir=args.log_dir,
            resume='allow',
            project=args.project,
            name=args.run_name,
            id=args.run_name,
            group=args.group,
            config=args
        )
        # wandb.log({'numel': numel})
    # record parameters
    args.device = args.device.type # just record type of device
    # calculate number of used gpus
    if args.device == 'cuda':
        args.num_gpus = torch.cuda.device_count()
    else:
        args.num_gpus = 0
    run_dir = os.path.join(args.log_dir, args.run_name)
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device # restore device
    train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_function(device)