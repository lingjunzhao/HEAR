#!/bin/bash --login
#SBATCH --job-name=train_airbert_calibrate_gpt
#SBATCH --output=slurm_logs/train_hallucination_type_gpt.out
#SBATCH --error=slurm_logs/train_hallucination_type_gpt.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=120gb
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4

set -x
module add cuda/8.0.44 cudnn/v5.1

name=train_hallucination_type_gpt
flag="--from_pretrained model-zoos/airbert-r2rRSA.bin
      --calibration_train cal_data/perturbation_train_intrinsic_extrinsic.json
      --calibration_val_seen cal_data/perturbation_val_seen_intrinsic_extrinsic.json
      --calibration_val_unseen cal_data/perturbation_val_unseen_intrinsic_extrinsic.json
      --learning_rate 1e-5
      --batch_size 128
      --val_batch_size 32
      --num_beams_train 2
      --num_beams 30
      --calibrate True
      --save_name $name 
     "

mkdir -p snap/$name

python3 -u \
-m torch.distributed.launch \
--nproc_per_node=4 \
--nnodes=1 \
--node_rank=0 \
 hear/train.py $flag

