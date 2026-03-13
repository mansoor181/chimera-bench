#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import time
import json
import argparse
import torch
import os.path as osp
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
setup_seed(SEED)

from data.dataset import E2EDataset, VOCAB, BaseData
from pytorch_lightning import Trainer
import pytorch_lightning.loggers as plog

from models import dyMEANITF, dyMEANOptITF, dyAbITF, dyAbOptITF
from models.callbacks import SetupCallback, BestCheckpointCallback, EpochEndCallback

from torch import distributed as dist
def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def parse():
    parser = argparse.ArgumentParser(description='training')
    # task
    parser.add_argument('--task', type=str, default='single_cdr_design', choices=['single_cdr_design', 'multi_cdr_design', 'single_cdr_opt', 'multi_cdr_opt', 'struct_prediction', 'full_design'])
    parser.add_argument('--ex_name', type=str, default='DEBUG')
    parser.add_argument('--strategy', type=str, default='ddp')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--wandb_offline', type=int, default=0)
    # data
    parser.add_argument('--train_set', type=str, help='path to train set')
    parser.add_argument('--valid_set', type=str, help='path to valid set')
    parser.add_argument('--test_set', type=str, help='path to valid set')
    parser.add_argument('--cdr', type=str, default=None, nargs='+', help='cdr to generate, L1/2/3, H1/2/3,(can be list, e.g., L3 H3) None for all including framework')
    parser.add_argument('--paratope', type=str, default='H3', nargs='+', help='cdrs to use as paratope')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='exponential decay from lr to final_lr')
    parser.add_argument('--warmup', type=int, default=0, help='linear learning rate warmup')
    parser.add_argument('--max_epochs', type=int, default=None, help='max training epoch')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, default='./results', help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--patience', type=int, default=1000, help='patience before early stopping (set with a large number to turn off early stopping)')
    parser.add_argument('--save_topk', type=int, default=5, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=4)

    # device
    parser.add_argument('--gpus', type=int, nargs='+', default=[1], help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # model
    parser.add_argument('--model_type', type=str, choices=['dyMEAN', 'dyMEANOpt', 'dyAb', 'dyAbOpt'],
                        help='Type of model')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of residue/atom embedding')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--k_neighbors', type=int, default=9, help='Number of neighbors in KNN graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--iter_round', type=int, default=3, help='Number of iterations for generation')

    # loss
    parser.add_argument('--weight_dsm', type=float, default=1.0)

    # dyMEANOpt related
    parser.add_argument('--seq_warmup', type=int, default=0, help='Number of epochs before starting training sequence')

    # task setting
    parser.add_argument('--struct_only', action='store_true', help='Predict complex structure given the sequence')
    parser.add_argument('--bind_dist_cutoff', type=float, default=6.6, help='distance cutoff to decide the binding interface')

    # ablation
    parser.add_argument('--no_pred_edge_dist', action='store_true', help='Turn off edge distance prediction at the interface')
    parser.add_argument('--backbone_only', action='store_true', help='Model backbone only')
    parser.add_argument('--fix_channel_weights', action='store_true', help='Fix channel weights, may also for special use (e.g. antigen with modified AAs)')
    parser.add_argument('--no_memory', action='store_true', help='No memory passing')

    parser.add_argument('--flexible', type=int, default=0)
    parser.add_argument('--module_type', type=int, default=0)
    parser.add_argument('--coord_eps', type=float, default=5e-4)

    return parser.parse_args()


def main(args):
    ########## define your model/trainer/trainconfig #########
    config = args.__dict__

    ########### load your data ###########
    local_rank, _ = get_dist_info()
    if (len(args.gpus) > 1 and int(local_rank) == 0) or len(args.gpus) == 1:
        print_log(args)
        print_log(f'CDR type: {args.cdr}')
        print_log(f'Paratope: {args.paratope}')
        print_log('structure only' if args.struct_only else 'sequence & structure codesign')

    train_set = E2EDataset(args.train_set, cdr=args.cdr, paratope=args.paratope, full_antigen=False, use_af2ag=args.flexible)
    valid_set = E2EDataset(args.valid_set, cdr=args.cdr, paratope=args.paratope, full_antigen=False, use_af2ag=args.flexible)
    test_set = E2EDataset(args.test_set, cdr=args.cdr, full_antigen=False, use_af2ag=args.flexible)
    collate_fn = E2EDataset.collate_fn

    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    config['step_per_epoch'] = step_per_epoch
    if args.local_rank == 0 or args.local_rank == -1:
        print_log(f'step per epoch: {step_per_epoch}')

    if len(args.gpus) > 1:
        args.local_rank = local_rank
        args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        args.local_rank = -1
    config['local_rank'] = args.local_rank

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=args.shuffle, pin_memory=True,
                    collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                    num_workers=args.num_workers, pin_memory=True,
                    collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size * 2,
                    num_workers=args.num_workers, pin_memory=True,
                    collate_fn=collate_fn)
    data_itf = BaseData(train_loader, valid_loader, test_loader)
    args.test_data = test_set.data

    ########## define your model ##########
    if args.model_type == 'dyMEAN':
        model_itf = dyMEANITF(**vars(args))
    elif args.model_type == 'dyMEANOpt':
        model_itf = dyMEANOptITF(**vars(args))
    elif args.model_type == 'dyAb':
        model_itf = dyAbITF(**vars(args))
    elif args.model_type == 'dyAbOpt':
        model_itf = dyAbOptITF(**vars(args))
    else:
        raise NotImplemented(f'model {args.model_type} not implemented')
    
    save_dir = osp.join(args.save_dir, args.ex_name)
    ckpt_dir = osp.join(save_dir, 'checkpoints')

    ########## prepare your callbacks ##########
    setup_callback = SetupCallback(
        prefix=args.task,
        setup_time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
        save_dir=save_dir,
        ckpt_dir=ckpt_dir,
        args=args,
        argv_content=sys.argv + ["gpus: {}".format(torch.cuda.device_count())],
    )
    ckpt_callback = BestCheckpointCallback(
        monitor='valid_loss',
        filename='{epoch:02d}_{step}_{valid_loss:.3f}',
        mode='min',
        save_last=True,
        save_top_k=args.save_topk,
        dirpath=ckpt_dir,
        verbose=True,
    )
    epochend_callback = EpochEndCallback()
    callbacks = [setup_callback, ckpt_callback, epochend_callback]
    # callbacks = []

    ########## training ##########
    logger = plog.WandbLogger(project='dyAb', name=args.ex_name, save_dir=save_dir, config=config, offline=args.wandb_offline)
    # logger = None
    trainer = Trainer(accelerator='gpu', strategy='auto', callbacks=callbacks,
        max_epochs=args.max_epochs, max_steps=args.max_epochs * step_per_epoch, devices=args.gpus, gradient_clip_val=args.gradient_clip_val, gradient_clip_algorithm='norm', logger=logger)
    trainer.fit(model_itf, data_itf)

    ########## evaluating ##########
    os.environ['OPENMM_CPU_THREADS'] = '1'
    trainer = Trainer(accelerator='gpu', callbacks=callbacks, devices=[0], logger=logger)
    trainer.test(model_itf, data_itf, ckpt_path=osp.join(ckpt_dir, 'best.ckpt'))

    metrics = model_itf.cal_metric()
    with open(osp.join(save_dir, 'metrics.json'), 'w') as file_obj:
        json.dump(metrics, file_obj)

if __name__ == '__main__':
    args = parse()
    configfile = osp.join('scripts/train/configs', args.task + '.json')

    with open(configfile, 'r') as file:
        config = json.load(file)
    
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.max_epochs is not None:
        config['max_epochs'] = args.max_epochs
        
    args.__dict__.update(config)

    main(args)