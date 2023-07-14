NUM_GPU=4
GPU_IDS="0,1,2,3"
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
model_name_or_path="microsoft/git-base-coco"
CUDA_VISIBLE_DEVICES=$GPU_IDS \
torchrun --nproc_per_node $NUM_GPU train.py \
    --output_dir "output" \
    --seed 42 \
    --model_name_or_path ${model_name_or_path} \
    --train_data_path "data/preprocess_train.csv" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --logging_strategy "steps" \
    --logging_steps "10" \
    --eval_steps "250" \
    --save_steps "250" \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --dataloader_num_workers "4" \
    --label_names "labels" \
    --fp16 \
    --remove_unused_columns "False"