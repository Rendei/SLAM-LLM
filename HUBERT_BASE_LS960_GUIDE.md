# Using hubert-base-ls960 as Encoder for ASR

This guide shows you how to use the HuggingFace `hubert-base-ls960` model as an encoder for ASR tasks in SLAM-LLM.

## Quick Start

The framework now supports HuggingFace Hubert models! You can use `facebook/hubert-base-ls960` directly.

## Step 1: Install Dependencies

Make sure you have `transformers` installed:

```bash
pip install transformers
```

## Step 2: Configuration

### Option A: Using HuggingFace Model Identifier

Update your training script or config to use the HuggingFace Hubert encoder:

```python
# In your config or command line
model_config.encoder_name = "hubert_hf"  # or "hf_hubert"
model_config.encoder_path = "facebook/hubert-base-ls960"  # HuggingFace model ID
model_config.encoder_dim = 768  # Hidden size for hubert-base-ls960
model_config.encoder_projector = "linear"
dataset_config.input_type = "raw"
dataset_config.normalize = False  # HuggingFace Hubert handles normalization internally
```

### Option B: Using Local Model Path

If you've downloaded the model locally:

```python
model_config.encoder_name = "hubert_hf"
model_config.encoder_path = "/path/to/local/hubert-base-ls960"
model_config.encoder_dim = 768
```

## Step 3: Training Command

### Example Training Script

Create a training script (e.g., `finetune_hubert_base_ls960.sh`):

```bash
#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

run_dir=/path/to/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

# Model paths
speech_encoder_path="facebook/hubert-base-ls960"  # HuggingFace model ID
llm_path=/path/to/vicuna-7b-v1.5
train_data_path=/path/to/librispeech_train_960h.jsonl
val_data_path=/path/to/librispeech_dev_other.jsonl

output_dir=/path/to/output/hubert-base-ls960-$(date +"%Y%m%d")

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=hubert_hf \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=768 \
++model_config.encoder_projector=linear \
++model_config.encoder_projector_ds_rate=5 \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=raw \
++dataset_config.normalize=false \
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
```

### Direct Python Command

```bash
python examples/asr_librispeech/finetune_asr.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++model_config.encoder_name=hubert_hf \
    ++model_config.encoder_path=facebook/hubert-base-ls960 \
    ++model_config.encoder_dim=768 \
    ++model_config.encoder_projector=linear \
    ++dataset_config.input_type=raw \
    ++dataset_config.normalize=false \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.batch_size_training=4
```

## Step 4: Inference

For inference, use the same encoder configuration:

```bash
python examples/asr_librispeech/inference_asr_batch.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++model_config.encoder_name=hubert_hf \
    ++model_config.encoder_path=facebook/hubert-base-ls960 \
    ++model_config.encoder_dim=768 \
    ++model_config.encoder_projector=linear \
    ++dataset_config.input_type=raw \
    ++dataset_config.normalize=false \
    ++ckpt_path=/path/to/checkpoint.pt
```

## Key Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `encoder_name` | `"hubert_hf"` or `"hf_hubert"` | Identifier for HuggingFace Hubert encoder |
| `encoder_path` | `"facebook/hubert-base-ls960"` | HuggingFace model ID or local path |
| `encoder_dim` | `768` | Hidden dimension of hubert-base-ls960 |
| `encoder_projector` | `"linear"` | Projector type (linear, q-former, cov1d-linear) |
| `input_type` | `"raw"` | Input format (raw audio, not mel) |
| `normalize` | `false` | HuggingFace models handle normalization internally |

## Model Information

- **Model**: `facebook/hubert-base-ls960`
- **Architecture**: Hubert (Base)
- **Training Data**: LibriSpeech 960 hours
- **Hidden Size**: 768
- **Sample Rate**: 16kHz
- **Input Format**: Raw waveform (1D tensor)

## Differences from Fairseq Hubert

The HuggingFace Hubert encoder (`hubert_hf`) differs from the Fairseq version (`hubert`):

1. **Loading**: Uses `transformers` library instead of `fairseq`
2. **Interface**: Uses `extract_features()` method instead of direct forward call
3. **Output**: Returns features directly without needing `encoder_type` configuration
4. **Normalization**: Handled internally by the model

## Troubleshooting

### Issue: Model not found
**Solution**: Make sure you have internet connection for HuggingFace to download the model, or download it locally first:
```python
from transformers import HubertModel
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
# Save locally if needed
```

### Issue: Dimension mismatch
**Solution**: Ensure `encoder_dim=768` matches the model's hidden size

### Issue: CUDA out of memory
**Solution**: 
- Reduce `batch_size_training`
- Enable gradient checkpointing
- Use mixed precision training (`use_fp16=true`)

## Example: Complete Config File

You can also create a custom config file:

```yaml
# conf/hubert_base_ls960.yaml
defaults:
  - prompt

model_config:
  encoder_name: hubert_hf
  encoder_path: facebook/hubert-base-ls960
  encoder_dim: 768
  encoder_projector: linear
  encoder_projector_ds_rate: 5

dataset_config:
  input_type: raw
  normalize: false

train_config:
  freeze_encoder: true
  freeze_llm: true
  batch_size_training: 4
```

Then use it:
```bash
python examples/asr_librispeech/finetune_asr.py \
    --config-path "conf" \
    --config-name "hubert_base_ls960.yaml" \
    ++model_config.llm_path=/path/to/llm \
    ++dataset_config.train_data_path=/path/to/train.jsonl
```

## Additional Hubert Models

You can also use other HuggingFace Hubert models:
- `facebook/hubert-large-ls960` (encoder_dim: 1024)
- `facebook/hubert-xlarge-ls960` (encoder_dim: 1280)

Just adjust the `encoder_dim` accordingly!
