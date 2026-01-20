#!/bin/bash
# Example script for training with hubert-base-ls960 (HuggingFace)
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/root/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

# Using HuggingFace model identifier
speech_encoder_path="facebook/hubert-base-ls960"  # HuggingFace model ID
llm_path=/path/to/vicuna-7b-v1.5
train_data_path=/path/to/librispeech_train_960h.jsonl
val_data_path=/path/to/librispeech_dev_other.jsonl

output_dir=/root/tmp/vicuna-7b-v1.5-librispeech-linear-hubert-base-ls960-$(date +"%Y%m%d")

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=hubert_hf \
++model_config.normalize=false \
++dataset_config.normalize=false \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=768 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=raw \
++train_config.model_name=asr \
++train_config.num_epochs=3 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=2000 \
++train_config.batch_size_training=6 \
++train_config.val_batch_size=6 \
++train_config.num_workers_dataloader=0 \
++train_config.output_dir=$output_dir \
++metric=acc \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=true \
        $hydra_args
fi
