#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import time
from copy import deepcopy
import subprocess
from data.pdb_utils import Peptide, Protein, merge_to_one_chain
from configs import CACHE_DIR
from data.pdb_utils import Protein, AgAbComplex, merge_to_one_chain, merge_to_one_protein

# def exec_bin(mod_pdb, ref_pdb, log, backbone_only):
#     options = '-x'
#     if backbone_only:
#         options += ' -c'
#     cmd = f'lddt {options} {mod_pdb} {ref_pdb} > {log} 2>&1'
#     return os.system(cmd)


def exec_bin(mod_pdb, ref_pdb, log, backbone_only):
    options = '-x'
    if backbone_only:
        options += ' -c'
    cmd = f'lddt {options} {mod_pdb} {ref_pdb}'
    
    with open(log, 'w') as log_file:
        result = subprocess.run(
            cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ
        )
    return result.returncode


def lddt(mod_protein: Protein, ref_protein: Protein, backbone_only=False):
    # concatenate all chains to one chain
    mod_protein = merge_to_one_chain(mod_protein)
    ref_protein = merge_to_one_chain(ref_protein)

    mod_sign, ref_sign = id(mod_protein), id(ref_protein)
    mod_pdb = os.path.join(CACHE_DIR, f'lddt_{mod_sign}_mod_{time.time()}.pdb')
    ref_pdb = os.path.join(CACHE_DIR, f'lddt_{ref_sign}_ref_{time.time()}.pdb')
    log = os.path.join(CACHE_DIR, f'lddt_log_{mod_sign}_{ref_sign}.txt')
    
    mod_protein.to_pdb(mod_pdb)
    ref_protein.to_pdb(ref_pdb)

    res_code = exec_bin(mod_pdb, ref_pdb, log, backbone_only)
    if res_code != 0:
        raise ValueError(f'lddt execution failed')
    with open(log, 'r') as fin:
        text = fin.read()
    res = re.search(r'Global LDDT score: ([0-1]\.?[0-9]*)', text)
    score = float(res.group(1))
    os.remove(mod_pdb)
    os.remove(ref_pdb)
    os.remove(log)
    return score, text

def lddt_full(mod_complex: AgAbComplex, ref_complex: AgAbComplex, backbone_only=False):
    # Merge antigen and antibody chains into a single protein for each complex
    mod_antigen = merge_to_one_chain(mod_complex.antigen)
    ref_antigen = merge_to_one_chain(ref_complex.antigen)

    mod_antibody = merge_to_one_chain(mod_complex.antibody)
    ref_antibody = merge_to_one_chain(ref_complex.antibody)
    
    # Combine merged antigen and antibody into one protein
    mod_protein = merge_to_one_protein(mod_antigen, mod_antibody)
    ref_protein = merge_to_one_protein(ref_antigen, ref_antibody)

    # Generate unique identifiers based on memory addresses, add current timestamp for filename uniqueness
    mod_sign, ref_sign = id(mod_antigen), id(ref_antigen)
    mod_pdb = os.path.join(CACHE_DIR, f'lddt_{mod_sign}_mod_{time.time()}.pdb')
    ref_pdb = os.path.join(CACHE_DIR, f'lddt_{ref_sign}_ref_{time.time()}.pdb')
    log = os.path.join(CACHE_DIR, f'lddt_log_{mod_sign}_{ref_sign}.txt')

    # Save the proteins as PDB files
    mod_protein.to_pdb(mod_pdb)
    ref_protein.to_pdb(ref_pdb)

    # Execute LDDT binary with the pdb files, logging to file
    res_code = exec_bin(mod_pdb, ref_pdb, log, backbone_only)
    if res_code != 0:
        raise ValueError('lddt execution failed')
    
    # Read the log file to extract the global LDDT score
    with open(log, 'r') as fin:
        text = fin.read()
    res = re.search(r'Global LDDT score: ([0-1]\.?[0-9]*)', text)
    score = float(res.group(1))

    # Cleanup temporary files
    os.remove(mod_pdb)
    os.remove(ref_pdb)
    os.remove(log)

    return score, text
