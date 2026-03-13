import torch
import json
import os.path as osp
from tqdm import tqdm
import numpy as np

from .dyMEANOpt_itf import dyMEANOptITF
from .dyMEAN.dyAbOpt_model import dyAbOptModel
from data import VOCAB

from api.optimize import optimize, ComplexSummary
from evaluation.pred_ddg import pred_ddg, foldx_ddg, foldx_minimize_energy
from utils.relax import openmm_relax
from utils.logger import print_log


class dyAbOptITF(dyMEANOptITF):
    def __init__(self, **kwargs):
        super(dyAbOptITF, self).__init__()
        self.save_hyperparameters()
        self.res_dir = osp.join(self.hparams.save_dir, self.hparams.ex_name, 'results')
        self.writer = None  # initialize right before training
        self.writer_buffer = {}

        self.use_foldx = False
        self.cdr_type = 'H3'
        self.n_samples = 100
        self.num_residue_changes = 0
        self.num_optimize_steps = 50
        self.batch_size = 32
        self.num_workers = 4

        self.model = dyAbOptModel(self.hparams.embed_dim, self.hparams.hidden_size, VOCAB.MAX_ATOM_NUMBER,
            VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
            self.hparams.k_neighbors, bind_dist_cutoff=self.hparams.bind_dist_cutoff,
            n_layers=self.hparams.n_layers, struct_only=self.hparams.struct_only,
            fix_atom_weights=self.hparams.fix_channel_weights, cdr_type=self.hparams.cdr)
        
    def test_step(self, batch, batch_idx):
        return 
        
    def cal_metric(self,):
        predictor = torch.load('checkpoints/cdrh3_ddg_predictor.ckpt', map_location='cpu')
        predictor.to('cuda:0')
        predictor.eval()

        with open(osp.join('all_data/SKEMPI/test.json'), 'r') as fin:
            items = [json.loads(line) for line in fin.read().strip().split('\n')]

        log = open(osp.join(self.res_dir, 'log.txt'), 'w')
        best_scores, success = [], []
        changes = []
        for item_id, item in enumerate(items):
            summary = ComplexSummary(
                pdb=item['pdb_data_path'],
                heavy_chain=item['heavy_chain'],
                light_chain=item['light_chain'],
                antigen_chains=item['antigen_chains']
            )
            pdb_id = item['pdb']
            out_dir = osp.join(self.res_dir, pdb_id)
            print_log(f'Optimizing {pdb_id}, {item_id + 1} / {len(items)}')
            gen_pdbs, gen_cdrs = optimize(
                ckpt=self.model,
                predictor_ckpt=predictor,
                gpu=0,
                cplx_summary=summary,
                num_residue_changes=[self.num_residue_changes for _ in range(self.n_samples)],
                out_dir=out_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                enable_openmm_relax=False,  # for fast evaluation
                optimize_steps=self.num_optimize_steps,
            )

            ori_cdr, ori_pdb, scores = item[f'cdr{self.cdr_type.lower()}_seq'], summary.pdb, []

            item_log = open(osp.join(out_dir, 'detail.txt'), 'w')
            different_cnt, cur_changes = 0, []
            for gen_pdb, gen_cdr in tqdm(zip(gen_pdbs, gen_cdrs), total=len(gen_pdbs)):
                change_cnt = 0
                if gen_cdr != ori_cdr:
                    if self.use_foldx:
                        gen_pdb = openmm_relax(gen_pdb, gen_pdb)
                        gen_pdb = foldx_minimize_energy(gen_pdb)
                        try:
                            score = foldx_ddg(ori_pdb, gen_pdb, summary.antigen_chains, [summary.heavy_chain, summary.light_chain])
                        except ValueError as e:
                            print(e)
                            score = 0
                    else:
                        score = pred_ddg(ori_pdb, gen_pdb)
                    # inputs.append((gen_pdb, summary, ori_dg, interface))
                    different_cnt += 1
                    for a, b in zip(gen_cdr, ori_cdr):
                        if a != b:
                            change_cnt += 1
                else:
                    # continue
                    score = 0
                scores.append(score)
                cur_changes.append(change_cnt)

            avg_change = sum(cur_changes) / different_cnt
            print_log(f'obtained {different_cnt} candidates, average change {avg_change}')
            
            sucess_rate = sum(1 if s < 0 else 0 for s in scores) / len(scores)
            success.append(sucess_rate)
            mean_score = round(np.mean(scores), 3)
            best_score_idx = min([k for k in range(len(scores))], key=lambda k: scores[k])
            best_scores.append(scores[best_score_idx])
            changes.append(cur_changes[best_score_idx])
            message = f'{pdb_id}: mean ddg {mean_score}, best ddg {round(scores[best_score_idx], 3)}, diff cnt {different_cnt}, success rate {sucess_rate}, change: {cur_changes[best_score_idx]}, sample {gen_pdbs[best_score_idx]}\n'
            item_log.write(message)
            item_log.close()
            
            log.write(message)
            log.flush()
            
            print_log(message)

        final_message = f'average best scores: {np.mean(best_scores)}, IMP: {np.mean(success)}, changes: {np.mean(changes)}'
        print_log(final_message)
        log.write(final_message)
        log.close()