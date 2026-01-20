import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):
        
        def extract_variable_length_features(self, x: torch.Tensor):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            # x = (x + self.positional_embedding).to(x.dtype)
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            for block in self.blocks:
                x = block(x)

            x = self.ln_post(x)
            return x

        if model_config.whisper_decode:
            import whisper
            whisper_model = whisper.load_model(name=model_config.encoder_path, device='cpu')
            whisper_model.encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, whisper_model.encoder)
            return whisper_model

        if model_config.encoder_path_hf is not None:
            from transformers import WhisperModel
            encoder = WhisperModel.from_pretrained(model_config.encoder_path_hf,torch_dtype=torch.bfloat16).encoder
        else:
            import whisper
            encoder = whisper.load_model(name=model_config.encoder_path, device='cpu').encoder
            encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)
        return encoder


class BEATsEncoder:

    @classmethod
    def load(cls, model_config):
        from .BEATs.BEATs import BEATs, BEATsConfig
        checkpoint = torch.load(model_config.encoder_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])

        return BEATs_model


@dataclass
class UserDirModule:
    user_dir: str
    
class EATEncoder:
    
    @classmethod
    def load(cls, model_config):
        import fairseq
        model_path = UserDirModule(model_config.encoder_fairseq_dir)
        fairseq.utils.import_user_module(model_path)
        EATEncoder, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        EATEncoder = EATEncoder[0]

        return EATEncoder
    
    def extract_features(self, source, padding_mask):
        return self.model.extract_features(source, padding_mask = padding_mask, mask=False, remove_extra_tokens = False)['x']

class CLAPEncoder: 

    @classmethod
    def load(cls, model_config): 
        from .CLAP.ase_model import ASE
        import ruamel.yaml as yaml
        with open(model_config.clap_config, 'r') as f: 
            clap_config = yaml.safe_load(f)
        clap_config['pd_text_support'] = model_config.get("pd_text_support", None)
        model = ASE(clap_config)
        checkpoint = torch.load(model_config.encoder_path)['model']
        model.load_state_dict(checkpoint)
        return model
    
class SpatialASTEncoder:
    @classmethod
    def load(cls, model_config):
        from functools import partial
        from .SpatialAST import SpatialAST 
        binaural_encoder = SpatialAST.BinauralEncoder(
            num_classes=355, drop_path_rate=0.1, num_cls_tokens=3,
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        checkpoint = torch.load(model_config.encoder_ckpt, map_location='cpu')
        binaural_encoder.load_state_dict(checkpoint['model'], strict=False) 
        return binaural_encoder

class WavLMEncoder(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def load(cls, model_config):
        from .wavlm.WavLM import WavLM, WavLMConfig
        checkpoint = torch.load(model_config.encoder_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        WavLM_model = WavLM(cfg)
        WavLM_model.load_state_dict(checkpoint['model'])
        assert model_config.normalize == cfg.normalize, "normalize flag in config and model checkpoint do not match"
 
        return cls(cfg, WavLM_model)

    def extract_features(self, source, padding_mask):
        return self.model.extract_features(source, padding_mask)[0]

class AVHubertEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        from .avhubert import hubert_pretraining, hubert, hubert_asr
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]
        return model

class HubertEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]
        if model_config.encoder_type == "pretrain":
            pass
        elif model_config.encoder_type == "finetune":
            model.w2v_encoder.proj = None
            model.w2v_encoder.apply_mask = False
        else:
            assert model_config.encoder_type in ["pretrain", "finetune"], "input_type must be one of [pretrain, finetune]" 
        return model


class HfHubertEncoder(nn.Module):
    """
    HuggingFace Hubert encoder wrapper.
    Supports models like 'facebook/hubert-base-ls960' from HuggingFace.
    """
    
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
    
    @classmethod
    def load(cls, model_config):
        from transformers import HubertModel, AutoModel
        
        # Try to load as HubertModel first, fallback to AutoModel
        try:
            if hasattr(model_config, 'encoder_path_hf') and model_config.encoder_path_hf:
                model_path = model_config.encoder_path_hf
            else:
                model_path = model_config.encoder_path
            
            # Try loading as HubertModel
            try:
                model = HubertModel.from_pretrained(model_path)
            except:
                # Fallback to AutoModel
                model = AutoModel.from_pretrained(model_path)
            
            config = model.config
            
            return cls(model, config)
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace Hubert model from {model_config.encoder_path}: {e}")
    
    def extract_features(self, source, padding_mask=None):
        """
        Extract features from audio input.
        
        Args:
            source: Audio input tensor [batch_size, sequence_length]
            padding_mask: Padding mask (True/1 for padding, False/0 for valid)
        
        Returns:
            torch.Tensor: Encoded features [batch_size, sequence_length, hidden_dim]
        """
        # Convert padding mask to attention mask format for HuggingFace
        # HuggingFace expects: 1 for valid tokens, 0 for padding
        if padding_mask is not None:
            attention_mask = (1 - padding_mask).long()
        else:
            attention_mask = None
        
        # HuggingFace Hubert expects input_values
        outputs = self.model(input_values=source, attention_mask=attention_mask)
        
        return outputs.last_hidden_state


class HfTextEncoder:

    @classmethod
    def load(cls, model_config):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_config.encoder_path)
        return model

class MusicFMEncoder(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def load(cls, model_config):
        from .musicfm.model.musicfm_25hz import MusicFM25Hz
        model = MusicFM25Hz(
            stat_path = model_config.encoder_stat_path,
            model_path = model_config.encoder_path,
            w2v2_config_path = model_config.get('encoder_config_path', "facebook/wav2vec2-conformer-rope-large-960h-ft")
        )
        return cls(model_config, model)

    def extract_features(self, source, padding_mask=None):
        _, hidden_states = self.model.get_predictions(source)
        out = hidden_states[self.config.encoder_layer_idx]
        return out

class Emotion2vecEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        model_path = UserDirModule(model_config.encoder_fairseq_dir)
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = model[0]

        return model