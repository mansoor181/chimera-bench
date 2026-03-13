python opt_gen.py --ckpt ./all_data/SKEMPI/models_single_cdr_opt/dyAbOpt_v1/checkpoints/best.ckpt \
 --predictor_ckpt ./checkpoints/cdrh3_ddg_predictor.ckpt \
 --summary_json ./all_data/SKEMPI/test.json \
 --save_dir ./all_data/camera_ready/new_affi_opt_1/ \
  --num_residue_changes 1 \
  --num_optimize_steps 50
sleep 100

python opt_gen.py --ckpt ./all_data/SKEMPI/models_single_cdr_opt/dyAbOpt_v1/checkpoints/best.ckpt \
 --predictor_ckpt ./checkpoints/cdrh3_ddg_predictor.ckpt \
 --summary_json ./all_data/SKEMPI/test.json \
 --save_dir ./all_data/camera_ready/new_affi_opt_2/ \
  --num_residue_changes 2 \
  --num_optimize_steps 50 
sleep 100

python opt_gen.py --ckpt ./all_data/SKEMPI/models_single_cdr_opt/dyAbOpt_v1/checkpoints/best.ckpt \
 --predictor_ckpt ./checkpoints/cdrh3_ddg_predictor.ckpt \
 --summary_json ./all_data/SKEMPI/test.json \
 --save_dir ./all_data/camera_ready/new_affi_opt_4/ \
  --num_residue_changes 4 \
  --num_optimize_steps 50

sleep 100

python opt_gen.py --ckpt ./all_data/SKEMPI/models_single_cdr_opt/dyAbOpt_v1/checkpoints/best.ckpt \
 --predictor_ckpt ./checkpoints/cdrh3_ddg_predictor.ckpt \
 --summary_json ./all_data/SKEMPI/test.json \
 --save_dir ./all_data/camera_ready/new_affi_opt_8/ \
  --num_residue_changes 8 \
  --num_optimize_steps 50

