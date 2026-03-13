from random import random

from .dyMEAN_itf import dyMEANITF
from .dyMEAN.dyMEANOpt_model import dyMEANOptModel
from data import VOCAB
import numpy as np
import torch

class dyMEANOptITF(dyMEANITF):
    def __init__(self, **kwargs):
        super(dyMEANOptITF, self).__init__()
        self.save_hyperparameters()
        self.writer = None  # initialize right before training
        self.writer_buffer = {}

        self.model = dyMEANOptModel(self.hparams.embed_dim, self.hparams.hidden_size, VOCAB.MAX_ATOM_NUMBER,
            VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
            self.hparams.k_neighbors, bind_dist_cutoff=self.hparams.bind_dist_cutoff,
            n_layers=self.hparams.n_layers, struct_only=self.hparams.struct_only,
            fix_atom_weights=self.hparams.fix_channel_weights, cdr_type=self.hparams.cdr)
        
    def get_context_ratio(self):
        ratio = random() * 0.9
        return ratio
    
    def training_step(self, batch, batch_idx):
        batch['seq_alpha'] = 1.0 - 1.0 * self.trainer.current_epoch / self.trainer.max_epochs
        loss = self.share_step(batch, batch_idx, val=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch['lengths']))
        return loss

    def validation_step(self, batch, batch_idx):
        batch['seq_alpha'] = 1
        loss = self.share_step(batch, batch_idx, val=True)

        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            self.log(name, value, on_epoch=True, sync_dist=True, batch_size=len(batch['lengths']))
        self.writer_buffer = {}

        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=len(batch['lengths']))
        return loss
    
    def share_step(self, batch, batch_idx, val=False):
        del batch['paratope_mask']
        del batch['template']
        batch['context_ratio'] = self.get_context_ratio()
        loss, seq_detail, structure_detail, pdev_detail = self.model(**batch)
        snll, aar = seq_detail
        struct_loss, xloss, bond_loss, sc_bond_loss = structure_detail
        pdev_loss, prmsd_loss = pdev_detail

        log_type = 'Validation' if val else 'Train'

        self.log_info(f'Overall/Loss/{log_type}', loss, val)

        self.log_info(f'Seq/SNLL/{log_type}', snll, val)
        self.log_info(f'Seq/AAR/{log_type}', aar, val)

        self.log_info(f'Struct/StructLoss/{log_type}', struct_loss, val)
        self.log_info(f'Struct/XLoss/{log_type}', xloss, val)
        self.log_info(f'Struct/BondLoss/{log_type}', bond_loss, val)
        self.log_info(f'Struct/SidechainBondLoss/{log_type}', sc_bond_loss, val)

        if pdev_loss is not None:
            self.log_info(f'PDev/PDevLoss/{log_type}', pdev_loss, val)
            self.log_info(f'PDev/PRMSDLoss/{log_type}', prmsd_loss, val)

        if not val:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log_info('lr', lr, val)
            self.log_info('context_ratio', batch['context_ratio'], val)
            self.log_info('seq_alpha', batch['seq_alpha'], val)
        return loss
    
    def log_info(self, name, value, val=False):
        if isinstance(value, torch.Tensor):
            value = value.cpu().item()
        if val:
            if name not in self.writer_buffer:
                self.writer_buffer[name] = []
            self.writer_buffer[name].append(value)
        else:
            self.log(name, value, sync_dist=True)