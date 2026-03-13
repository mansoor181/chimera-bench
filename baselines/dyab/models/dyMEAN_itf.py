import json
from torch.optim.optimizer import Optimizer
import wandb
import torch
import pytorch_lightning as pl
import numpy as np
import os.path as osp
from tqdm import tqdm
import multiprocessing
from math import cos, pi, log, exp

from .utils import to_cplx, cal_metrics
from .dyMEAN.dyMEAN_model import dyMEANModel
from data import VOCAB


class dyMEANITF(pl.LightningModule):
    def __init__(self, **kwargs):
        super(dyMEANITF, self).__init__()
        self.save_hyperparameters()
        self.res_dir = osp.join(self.hparams.save_dir, self.hparams.ex_name, 'results')
        self.writer = None  # initialize right before training
        self.writer_buffer = {}
        
        self.test_idx = 0
        self.summary_items = []

        self.model = dyMEANModel(self.hparams.embed_dim, self.hparams.hidden_size, VOCAB.MAX_ATOM_NUMBER,
            VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
            self.hparams.k_neighbors, bind_dist_cutoff=self.hparams.bind_dist_cutoff,
            n_layers=self.hparams.n_layers, struct_only=self.hparams.struct_only,
            iter_round=self.hparams.iter_round,
            backbone_only=self.hparams.backbone_only,
            fix_channel_weights=self.hparams.fix_channel_weights,
            pred_edge_dist=not self.hparams.no_pred_edge_dist,
            keep_memory=not self.hparams.no_memory,
            cdr_type=self.hparams.cdr, paratope=self.hparams.paratope, flexible=self.hparams.flexible, weight_dsm=self.hparams.weight_dsm)
                
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        log_alpha = log(self.hparams.final_lr / self.hparams.lr) / self.trainer.max_steps
        lr_lambda = lambda step: exp(log_alpha * (step + 1))  # equal to alpha^{step}
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step"
            },
        }

    def get_context_ratio(self):
        step = self.trainer.global_step
        ratio = 0.5 * (cos(step / self.trainer.max_steps * pi) + 1) * 0.9  # scale to [0, 0.9]
        return ratio
    
    def training_step(self, batch, batch_idx):
        batch_size = len(batch['lengths'])
        batch['context_ratio'] = self.get_context_ratio()
        loss = self.share_step(batch, batch_idx, val=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch['lengths'])
        batch['context_ratio'] = 0
        loss = self.share_step(batch, batch_idx, val=True)

        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            self.log(name, value, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.writer_buffer = {}

        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        return loss
    
    def share_step(self, batch, batch_idx, val=False):
        loss, seq_detail, structure_detail, dock_detail, pdev_detail = self.model(**batch)
        snll, aar = seq_detail
        struct_loss, xloss, bond_loss, sc_bond_loss = structure_detail
        dock_loss, interface_loss, ed_loss, r_ed_losses = dock_detail
        pdev_loss, prmsd_loss = pdev_detail

        log_type = 'Validation' if val else 'Train'

        self.log_info(f'Overall/Loss/{log_type}', loss, val)

        self.log_info(f'Seq/SNLL/{log_type}', snll, val)
        self.log_info(f'Seq/AAR/{log_type}', aar, val)

        self.log_info(f'Struct/StructLoss/{log_type}', struct_loss, val)
        self.log_info(f'Struct/XLoss/{log_type}', xloss, val)
        self.log_info(f'Struct/BondLoss/{log_type}', bond_loss, val)
        self.log_info(f'Struct/SidechainBondLoss/{log_type}', sc_bond_loss, val)

        self.log_info(f'Dock/DockLoss/{log_type}', dock_loss, val)
        self.log_info(f'Dock/SPLoss/{log_type}', interface_loss, val)
        self.log_info(f'Dock/EDLoss/{log_type}', ed_loss, val)
        for i, l in enumerate(r_ed_losses):
            self.log_info(f'Dock/edloss{i}/{log_type}', l, val)

        if pdev_loss is not None:
            self.log_info(f'PDev/PDevLoss/{log_type}', pdev_loss, val)
            self.log_info(f'PDev/PRMSDLoss/{log_type}', prmsd_loss, val)

        if not val:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log_info('lr', lr, val)
            self.log_info('context_ratio', batch['context_ratio'], val)
        return loss
    
    def test_step(self, batch, batch_idx):
        summary_items = []
        del batch['xloss_mask']
        X, S, pmets = self.model.sample(**batch)

        X, S, pmets = X.tolist(), S.tolist(), pmets.tolist()
        X_list, S_list = [], []
        cur_bid = -1
        if 'bid' in batch:
            batch_id = batch['bid']
        else:
            lengths = batch['lengths']
            batch_id = torch.zeros_like(batch['S'])  # [N]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        for i, bid in enumerate(batch_id):
            if bid != cur_bid:
                cur_bid = bid
                X_list.append([])
                S_list.append([])
            X_list[-1].append(X[i])
            S_list[-1].append(S[i])
        
        for i, (x, s) in enumerate(zip(X_list, S_list)):
            ori_cplx = self.hparams.test_data[self.test_idx]
            cplx = to_cplx(ori_cplx, x, s)
            pdb_id = cplx.get_id().split('(')[0]
            try:
                mod_pdb = osp.join(self.res_dir, pdb_id + '.pdb')
                cplx.to_pdb(mod_pdb)
                ref_pdb = osp.join(self.res_dir, pdb_id + '_original.pdb')
                ori_cplx.to_pdb(ref_pdb)
                summary_items.append({
                    'mod_pdb': mod_pdb,
                    'ref_pdb': ref_pdb,
                    'H': cplx.heavy_chain,
                    'L': cplx.light_chain,
                    'A': cplx.antigen.get_chain_names(),
                    'cdr_type': self.hparams.cdr,
                    'pdb': pdb_id,
                    'pmetric': pmets[i]
                })
                self.test_idx += 1
            except:
                print(pdb_id, self.test_idx)
                continue
        self.summary_items.extend(summary_items)
        return {'summary_items': summary_items}
    
    def on_test_end(self):
        summary_file = osp.join(self.res_dir, 'summary.json')
        with open(summary_file, 'w') as fout:
            fout.writelines(list(map(lambda item: json.dumps(item) + '\n', self.summary_items)))
        print(f'Summary of generated complexes written to {summary_file}')

    def cal_metric(self):
        metric_inputs, pdbs = [], [item['pdb'] for item in self.summary_items]
        pmets = []
        for item in self.summary_items:
            keys = ['mod_pdb', 'ref_pdb', 'H', 'L', 'A', 'cdr_type']
            inputs = [item[key] for key in keys]
            if 'sidechain' in item:
                inputs.append(item['sidechain'])
            metric_inputs.append(inputs)
            pmets.append(item['pmetric'])

        if self.hparams.num_workers > 1:
            with multiprocessing.Pool(self.hparams.num_workers) as pool:
                metrics = pool.map(cal_metrics, metric_inputs)
        else:
            metrics = [cal_metrics(inputs) for inputs in tqdm(metric_inputs)]

        result_dict = {}
        for name in metrics[0]:
            vals = [item[name] for item in metrics]
            print(f'{name}: {sum(vals) / len(vals)}')
            tname = 'Test/' + name
            if self.trainer.is_global_zero:
                wandb.log({tname: sum(vals) / len(vals)})
            result_dict[tname] = sum(vals) / len(vals)
            lowest_i = min([i for i in range(len(vals))], key=lambda i: vals[i])
            highest_i = max([i for i in range(len(vals))], key=lambda i: vals[i])
            print(f'\tlowest: {vals[lowest_i]}, pdb: {pdbs[lowest_i]}')
            print(f'\thighest: {vals[highest_i]}, pdb: {pdbs[highest_i]}')
            # calculate correlation
            corr = np.corrcoef(pmets, vals)[0][1]
            print(f'\tpearson correlation with development metric: {corr}')
        return result_dict

    def log_info(self, name, value, val=False):
        if isinstance(value, torch.Tensor):
            value = value.cpu().item()
        if val:
            if name not in self.writer_buffer:
                self.writer_buffer[name] = []
            self.writer_buffer[name].append(value)
        else:
            self.log(name, value, sync_dist=True)