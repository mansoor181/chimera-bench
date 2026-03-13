python train.py --task single_cdr_design --gpus 0 --wandb_offline 1 --flexible 1 --model_type dyAb --ex_name dyAb_single_cdrh3_m1 --module_type 1 &
python train.py --task single_cdr_design --gpus 1 --wandb_offline 1 --flexible 1 --model_type dyAb --ex_name dyAb_single_cdrh3_m0  --module_type 0 &
