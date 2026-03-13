#!/usr/bin/env bash

trap 'kill 0' SIGINT SIGTERM

exec python train.py --task single_cdr_opt --gpus 1 --wandb_offline 1 --flexible 1 --model_type dyAbOpt --train_set ./all_data/SKEMPI/train.json --valid_set ./all_data/SKEMPI/valid.json --test_set all_data/SKEMPI/test.json --save_dir ./all_data/RAbD/single_cdr_opt/ --ex_name dyAbOpt_v1 --module_type 1&
