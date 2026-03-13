import os
import logging
import subprocess
import sys
import torch
from time import time
from collections import defaultdict
from data import VOCAB, Residue, Peptide, Protein, AgAbComplex

import numpy as np
from evaluation.rmsd import compute_rmsd
from evaluation.tm_score import tm_score
from evaluation.lddt import lddt,lddt_full
from evaluation.dockq import dockq
from utils.relax import openmm_relax, rosetta_sidechain_packing
from utils.logger import print_log

from configs import CONTACT_DIST

def collect_env():
    """Collect the information of the running environments."""
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        from torch.utils.cpp_extension import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    '"{}" -V | tail -n1'.format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            env_info['GPU ' + ','.join(devids)] = name

    gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    gcc = gcc.decode('utf-8').strip()
    env_info['GCC'] = gcc

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()

    return env_info


def print_log(message):
    print(message)
    logging.info(message)


def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    return path


def to_cplx(ori_cplx, ab_x, ab_s) -> AgAbComplex:
    heavy_chain, light_chain = [], []
    chain = None
    for residue, residue_x in zip(ab_s, ab_x):
        residue = VOCAB.idx_to_symbol(residue)
        if residue == VOCAB.BOA:
            continue
        elif residue == VOCAB.BOH:
            chain = heavy_chain
            continue
        elif residue == VOCAB.BOL:
            chain = light_chain
            continue
        if chain is None:  # still in antigen region
            continue
        coord, atoms = {}, VOCAB.backbone_atoms + VOCAB.get_sidechain_info(residue)

        for atom, x in zip(atoms, residue_x):
            coord[atom] = x
        chain.append(Residue(
            residue, coord, _id=(len(chain), ' ')
        ))
    heavy_chain = Peptide(ori_cplx.heavy_chain, heavy_chain)
    light_chain = Peptide(ori_cplx.light_chain, light_chain)
    for res, ori_res in zip(heavy_chain, ori_cplx.get_heavy_chain()):
        res.id = ori_res.id
    for res, ori_res in zip(light_chain, ori_cplx.get_light_chain()):
        res.id = ori_res.id

    peptides = {
        ori_cplx.heavy_chain: heavy_chain,
        ori_cplx.light_chain: light_chain
    }
    antibody = Protein(ori_cplx.pdb_id, peptides)
    cplx = AgAbComplex(
        ori_cplx.antigen, antibody, ori_cplx.heavy_chain,
        ori_cplx.light_chain, skip_epitope_cal=True,
        skip_validity_check=True
    )
    cplx.cdr_pos = ori_cplx.cdr_pos
    return cplx


def cal_metrics(inputs):
    if len(inputs) == 6:
        mod_pdb, ref_pdb, H, L, A, cdr_type = inputs
        sidechain = False
    elif len(inputs) == 7:
        mod_pdb, ref_pdb, H, L, A, cdr_type, sidechain = inputs
    do_refine = False

    # sidechain packing
    if sidechain:
        refined_pdb = mod_pdb[:-4] + '_sidechain.pdb'
        mod_pdb = rosetta_sidechain_packing(mod_pdb, refined_pdb)

    # load complex
    if do_refine:
        refined_pdb = mod_pdb[:-4] + '_refine.pdb'
        pdb_id = os.path.split(mod_pdb)[-1]
        print(f'{pdb_id} started refining')
        start = time()
        mod_pdb = openmm_relax(mod_pdb, refined_pdb, excluded_chains=A)  # relax clashes
        print(f'{pdb_id} finished openmm relax, elapsed {round(time() - start)} s')
    mod_cplx = AgAbComplex.from_pdb(mod_pdb, H, L, A, skip_epitope_cal=True)
    ref_cplx = AgAbComplex.from_pdb(ref_pdb, H, L, A, skip_epitope_cal=False)

    results = {}
    cdr_type = [cdr_type] if type(cdr_type) == str else cdr_type

    # 1. AAR & CAAR
    # CAAR
    epitope = ref_cplx.get_epitope()
    is_contact = []
    if cdr_type is None:  # entire antibody
        gt_s = ref_cplx.get_heavy_chain().get_seq() + ref_cplx.get_light_chain().get_seq()
        pred_s = mod_cplx.get_heavy_chain().get_seq() + mod_cplx.get_light_chain().get_seq()
        # contact
        for chain in [ref_cplx.get_heavy_chain(), ref_cplx.get_light_chain()]:
            for ab_residue in chain:
                contact = False
                for ag_residue, _, _ in epitope:
                    dist = ab_residue.dist_to(ag_residue)
                    if dist < CONTACT_DIST:
                        contact = True
                is_contact.append(int(contact))
    else:
        gt_s, pred_s = '', ''
        for cdr in cdr_type:
            gt_cdr = ref_cplx.get_cdr(cdr)
            cur_gt_s = gt_cdr.get_seq()
            cur_pred_s = mod_cplx.get_cdr(cdr).get_seq()
            gt_s += cur_gt_s
            pred_s += cur_pred_s
            # contact
            cur_contact = []
            for ab_residue in gt_cdr:
                contact = False
                for ag_residue, _, _ in epitope:
                    dist = ab_residue.dist_to(ag_residue)
                    if dist < CONTACT_DIST:
                        contact = True
                cur_contact.append(int(contact))
            is_contact.extend(cur_contact)

            hit, chit = 0, 0
            for a, b, contact in zip(cur_gt_s, cur_pred_s, cur_contact):
                if a == b:
                    hit += 1
                    if contact == 1:
                        chit += 1
            results[f'AAR {cdr}'] = hit * 1.0 / len(cur_gt_s)
            results[f'CAAR {cdr}'] = chit * 1.0 / (sum(cur_contact) + 1e-10)

    if len(gt_s) != len(pred_s):
        print_log(f'Length conflict: {len(gt_s)} and {len(pred_s)}', level='WARN')
    hit, chit = 0, 0
    for a, b, contact in zip(gt_s, pred_s, is_contact):
        if a == b:
            hit += 1
            if contact == 1:
                chit += 1
    results['AAR'] = hit * 1.0 / len(gt_s)
    results['CAAR'] = chit * 1.0 / (sum(is_contact) + 1e-10)

    # 2. RMSD(CA) w/o align
    gt_x, pred_x = [], []
    for xl, c in zip([gt_x, pred_x], [ref_cplx, mod_cplx]):
        for chain in [c.get_heavy_chain(), c.get_light_chain()]:
            for i in range(len(chain)):
                xl.append(chain.get_ca_pos(i))
    assert len(gt_x) == len(pred_x), f'coordinates length conflict'
    gt_x, pred_x = np.array(gt_x), np.array(pred_x)
    results['RMSD(CA) aligned'] = compute_rmsd(gt_x, pred_x, aligned=False)
    results['RMSD(CA)'] = compute_rmsd(gt_x, pred_x, aligned=True)
    if cdr_type is not None:
        for cdr in cdr_type:
            gt_cdr, pred_cdr = ref_cplx.get_cdr(cdr), mod_cplx.get_cdr(cdr)
            gt_x = np.array([gt_cdr.get_ca_pos(i) for i in range(len(gt_cdr))])
            pred_x = np.array([pred_cdr.get_ca_pos(i) for i in range(len(pred_cdr))])
            results[f'RMSD(CA) CDR{cdr}'] = compute_rmsd(gt_x, pred_x, aligned=True)
            results[f'RMSD(CA) CDR{cdr} aligned'] = compute_rmsd(gt_x, pred_x, aligned=False)

    # 3. TMscore
    results['TMscore'] = tm_score(mod_cplx.antibody, ref_cplx.antibody)

    # 4. LDDT
    # score, _ = lddt(mod_cplx.antibody, ref_cplx.antibody)
    score, _ = lddt_full(mod_cplx, ref_cplx)
    results['LDDT'] = score

    # 5. DockQ
    try:
        score = dockq(mod_cplx, ref_cplx, cdrh3_only=True) # consistent with HERN
    except Exception as e:
        print_log(f'Error in dockq: {e}, set to 0', level='ERROR')
        score = 0
    results['DockQ'] = score

    print(f'{mod_cplx.get_id()}: {results}')

    return results