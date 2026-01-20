import torch
import os
import logging
from slam_llm.models.slam_model import (
    slam_model,
    setup_tokenizer,
    setup_encoder,
    setup_encoder_projector,
    setup_llm,
)
from slam_llm.utils.train_utils import print_model_size

logger = logging.getLogger(__name__)

def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    encoder = setup_encoder(train_config, model_config, **kwargs)

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # projector
    encoder_projector = setup_encoder_projector(
        train_config, model_config, **kwargs
    )
    model = slam_model_asr(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get(
        "ckpt_path", None
    )  # FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None:
        logger.info("loading other parts from: {}".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model, tokenizer


class slam_model_asr(slam_model):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__(
            encoder,
            llm,
            encoder_projector,
            tokenizer,
            train_config,
            model_config,
            **kwargs,
        )


    @torch.no_grad()
    def inference(
        self,
        wav_path=None,
        prompt=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        # inference for asr model

        device = kwargs.get("device", "cuda")
        if os.path.exists(wav_path):  # Audio-Text QA
            import whisper
            import soundfile as sf

            # Determine input type based on encoder
            input_type = getattr(self.dataset_config, "input_type", "raw")
            
            if input_type == "mel" or self.model_config.encoder_name == "whisper":
                # Handle mel spectrogram input (for Whisper)
                audio_raw = whisper.load_audio(wav_path)
                audio_raw = whisper.pad_or_trim(audio_raw)

                mel_size = getattr(
                    self.dataset_config, "mel_size", 80
                )  # 80 for large v1 and v2, 128 for large v3
                audio_mel = (
                    whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size)
                    .permute(1, 0)[None, :, :]
                    .to(device)
                )

                encoder_outs = self.encoder.extract_variable_length_features(
                    audio_mel.permute(0, 2, 1)
                )
            else:
                # Handle raw audio input (for hubert_hf, wavlm, etc.)
                audio_raw, sr = sf.read(wav_path)
                audio_raw = torch.from_numpy(audio_raw).float()
                
                # Resample to 16kHz if needed (Hubert expects 16kHz)
                if sr != 16000:
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    audio_raw = resampler(audio_raw.unsqueeze(0)).squeeze(0)
                
                # Normalize if required
                normalize = getattr(self.dataset_config, "normalize", False)
                if normalize:
                    audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
                
                # Add batch dimension and move to device
                audio_raw = audio_raw.unsqueeze(0).to(device)  # [1, seq_len]
                
                # Extract features using encoder
                if self.model_config.encoder_name == "hubert_hf" or self.model_config.encoder_name == "hf_hubert":
                    encoder_outs = self.encoder.extract_features(audio_raw, padding_mask=None)
                elif hasattr(self.encoder, "extract_features"):
                    encoder_outs = self.encoder.extract_features(audio_raw, padding_mask=None)
                else:
                    # Fallback for other encoders
                    encoder_outs = self.encoder(audio_raw)

            if self.model_config.encoder_projector == "q-former":
                audio_mel_post_mask = torch.ones(
                    encoder_outs.size()[:-1], dtype=torch.long
                ).to(encoder_outs.device)
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
        else:  # Text QA
            encoder_outs = torch.empty(
                1, 0, self.llm.model.embed_tokens.embedding_dim
            ).to(device)

        prompt = "USER: {}\n ASSISTANT:".format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(device)

        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(prompt_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(prompt_ids)
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(prompt_ids)

        inputs_embeds = torch.cat(
            (encoder_outs, inputs_embeds[None, :, :]), dim=1
        )  # [audio,prompt]

        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(
            inputs_embeds.device
        )

        # generate
        model_outputs = self.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

        return model_outputs
