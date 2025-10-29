export WANDB_API_KEY="0f2cb2e82161bace177966302bdba87bff2a2f97"

export WANDB_PROJECT="COMP4901B-Homework2"
RUNNAME="HW2_16"
MODELPATH="SmolLM2-135M"
DATAPATH="smol-smoltalk-6k.json"
MODEL_SIZE="0.6B"
OUTPUTPATH="ckpt"
DEVICES="0"  # e.g. 0,1,2,3
NUM_GPUS=1
TOTALBSZ=512
BSZPERDEV=1
GRADACC=$((TOTALBSZ / NUM_GPUS / BSZPERDEV))
export CUDA_VISIBLE_DEVICES=${DEVICES}
echo "Training model ${MODEL_SIZE} using ${NUM_GPUS} GPUs, ${BSZPERDEV} batch size per GPU, ${GRADACC} gradient accumulation steps"

# Single GPU training without DeepSpeed
python train_hw_parallel.py \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 5 \
    --save_strategy "epoch" \
    --save_steps 5 \
    --save_total_limit 6 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval True \
    --model_max_length 2304 \
    --lazy_preprocess True \
    --report_to "wandb" \
    --run_name ${RUNNAME} \
    --bf16 True \
    --flash_attn False \
    --dataloader_num_workers 2 \
    --preprocess_workers 2 \
    --max_rounds 5 