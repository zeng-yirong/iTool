#!/bin/bash
# bash run_mcts_dpo.sh 2>&1 | tee log.txt

output_dir="path_to_save"

# 如果路径不存在，则运行 python main.py
python main.py --model_path $model_path \
    --infer_dataset_path $data_path \
    --save_buffer_name $save_buffer_path \
    --save_preference_dir $save_perfer_data_path \
    --output_dir $output_dir \
    --num_gpus 8 \
    --num_actions 3 \
    --temperature 1.5 \
    --seed 42 \
    --step_limit 2 \
    --depth 3 \
    --max_simulation_steps 2 \
    --dpo_iters 1 \
    --mcts_iters 5 \
    --pref_loss simpo \
    --pref_beta 2.5 \
    --simpo_gamma 0.5 \
    --prefer_lr 1e-6

