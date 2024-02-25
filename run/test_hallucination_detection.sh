#!/bin/bash --login
#SBATCH --job-name=test_airbert_hallucination_detection
#SBATCH --output=slurm_logs/test_airbert_hallucination_detection.out
#SBATCH --error=slurm_logs/test_airbert_hallucination_detection.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa6000

set -x
module add cuda/8.0.44 cudnn/v5.1

name=test_hallucination_detection
flag="--from_pretrained data/runs/run-train_hallucination_detection_gpt/pytorch_model_best_unseen.bin
      --calibration_input_file cal_data/t5_val_seen_highlighted_phrase_alters_gpt4_direction_dev_test.json
      --batch_size 32
      --num_beams 155
      --save_name $name 
     "

mkdir -p snap/$name

python3 -u hear/test.py $flag

