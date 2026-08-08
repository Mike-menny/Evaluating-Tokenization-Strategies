"""Decoder-only MIDI Transformer conditioned on composer and genre tokens."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_VOCAB_WEIGHT_KEYS = ("token_embedding.weight", "lm_head.weight")


def adapt_checkpoint_state_dict(
    state: dict[str, torch.Tensor],
    *,
    ckpt_vocab_size: int,
    target_vocab_size: int,
) -> dict[str, torch.Tensor]:
    """Slice token embeddings when loading a larger-vocab checkpoint into a smaller mode.

    Token IDs are laid out as a prefix hierarchy (note -> velocity -> beat -> pedal),
    so every downstream tokenization mode is a row-prefix of larger modes.
    """
    if ckpt_vocab_size == target_vocab_size:
        return state
    if ckpt_vocab_size < target_vocab_size:
        raise ValueError(
            f"Checkpoint vocab_size={ckpt_vocab_size} is smaller than "
            f"target vocab_size={target_vocab_size}; cannot load without resizing."
        )
    out = dict(state)
    for key in _VOCAB_WEIGHT_KEYS:
        weight = out.get(key)
        if weight is not None and weight.shape[0] > target_vocab_size:
            out[key] = weight[:target_vocab_size].clone()
    return out


@dataclass
class ConditionalMidiConfig:
    vocab_size: int
    num_composers: int
    num_genres: int
    d_model: int = 768
    n_heads: int = 12
    num_layers: int = 24
    ffn_dim: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 8192
    cond_dim: int = 128
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tokenization_mode: str = "full"
    num_composer_ids: int = 0   # total embedding rows including pad/unknown
    num_genre_ids: int = 0


class _SelfAttention(nn.Module):
    def __init__(self, config: ConditionalMidiConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _heads, seq_len, _dim = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split(q)
        k = self._split(k)
        v = self._split(v)
        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        q_len = q.shape[2]
        # Right padding is only at the sequence tail; real causal queries cannot
        # attend to future pad tokens, and padded query losses are ignored.
        # This keeps training on the fused SDPA causal path instead of building
        # a huge explicit [B, T, T] mask.
        is_causal = past_key_value is None and q_len > 1

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        present = (k, v) if use_cache else None
        return self.out(self._merge(y)), present


class _CrossAttention(nn.Module):
    def __init__(self, config: ConditionalMidiConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q = nn.Linear(config.d_model, config.d_model)
        self.k = nn.Linear(config.d_model, config.d_model)
        self.v = nn.Linear(config.d_model, config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _heads, seq_len, _dim = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self._split(self.q(x))
        k = self._split(self.k(memory))
        v = self._split(self.v(memory))
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(self._merge(y))


class _DecoderLayer(nn.Module):
    def __init__(self, config: ConditionalMidiConfig):
        super().__init__()
        self.self_norm = nn.LayerNorm(config.d_model)
        self.cross_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.self_attn = _SelfAttention(config)
        self.cross_attn = _CrossAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        y, present = self.self_attn(
            self.self_norm(x),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + self.dropout(y)
        x = x + self.dropout(self.cross_attn(self.cross_norm(x), memory))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, present


class ConditionalMidiTransformer(nn.Module):
    def __init__(self, config: ConditionalMidiConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        # If num_composer_ids / num_genre_ids are provided they include pad+unknown
        # rows; otherwise fall back to the old num_composers + num_genres total.
        cond_vocab_size = (
            (self.config.num_composer_ids or self.config.num_composers)
            + (self.config.num_genre_ids or self.config.num_genres)
        )
        self.condition_embedding = nn.Embedding(cond_vocab_size, config.cond_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(config.cond_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.layers = nn.ModuleList([_DecoderLayer(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.gradient_checkpointing = False
        self._reset_parameters()

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def _checkpoint_layer(self, layer: _DecoderLayer, x: torch.Tensor, memory: torch.Tensor,
                          attention_mask: Optional[torch.Tensor], past_key_value, use_cache: bool):
        if self.gradient_checkpointing and self.training:
            def _custom_forward(x_, memory_, attention_mask_, past_key_value_, use_cache_):
                return layer(x_, memory=memory_, attention_mask=attention_mask_,
                             past_key_value=past_key_value_, use_cache=use_cache_)
            return torch.utils.checkpoint.checkpoint(
                _custom_forward, x, memory, attention_mask, past_key_value, use_cache,
                use_reentrant=False,
            )
        return layer(x, memory=memory, attention_mask=attention_mask,
                     past_key_value=past_key_value, use_cache=use_cache)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def _memory(self, composer_ids: torch.Tensor, genre_ids: torch.Tensor) -> torch.Tensor:
        genre_ids = genre_ids + self.config.num_composers
        cond_ids = torch.stack([composer_ids, genre_ids], dim=1)
        return self.condition_projection(self.condition_embedding(cond_ids))

    def forward(
        self,
        input_ids: torch.Tensor,
        composer_ids: torch.Tensor,
        genre_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if position_offset + seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {position_offset + seq_len} exceeds max_seq_len={self.config.max_seq_len}")

        positions = torch.arange(position_offset, position_offset + seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        memory = self._memory(composer_ids, genre_ids)

        next_key_values = [] if use_cache else None
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)  # type: ignore[list-item]
        hidden = x
        for layer, past in zip(self.layers, past_key_values):
            hidden, present = self._checkpoint_layer(
                layer, hidden, memory=memory, attention_mask=attention_mask,
                past_key_value=past, use_cache=use_cache,
            )
            if next_key_values is not None:
                next_key_values.append(present)
        logits = self.lm_head(self.final_norm(hidden))
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        if next_key_values is not None:
            out["past_key_values"] = next_key_values
        return out

    @torch.no_grad()
    def generate(
        self,
        composer_ids: torch.Tensor,
        genre_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.95,
        top_p: float = 0.98,
        logits_mask_fn=None,
        logits_processor=None,
    ) -> torch.Tensor:
        self.eval()
        device = composer_ids.device
        generated = torch.full(
            (composer_ids.shape[0], 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        finished = torch.zeros(composer_ids.shape[0], dtype=torch.bool, device=device)
        past_key_values = None
        for _ in range(max_new_tokens):
            if generated.shape[1] > self.config.max_seq_len:
                break
            step_input = generated[:, -1:]
            out = self(
                input_ids=step_input,
                composer_ids=composer_ids,
                genre_ids=genre_ids,
                past_key_values=past_key_values,
                use_cache=True,
                position_offset=generated.shape[1] - 1,
            )
            past_key_values = out["past_key_values"]
            logits = out["logits"][:, -1, :]
            if logits_processor is not None:
                logits = logits_processor.mask(logits)
            if logits_mask_fn is not None:
                logits = logits_mask_fn(logits, generated)
            next_token = _sample_top_p(logits, temperature=temperature, top_p=top_p)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.config.eos_token_id),
                next_token,
            )
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            if logits_processor is not None:
                logits_processor.update(next_token)
            finished |= next_token.eq(self.config.eos_token_id)
            if bool(finished.all()):
                break
        return generated

    def save_pretrained(self, output_dir: str | Path, extra_config: Optional[dict] = None) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.config)
        if extra_config:
            payload.update(extra_config)
        weights_path = output / "pytorch_model.bin"
        torch.save(self.state_dict(), weights_path)
        if not weights_path.exists() or weights_path.stat().st_size == 0:
            raise RuntimeError(f"Failed to save model weights to {weights_path}")
        (output / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, map_location: str | torch.device = "cpu"):
        model_dir = Path(model_dir)
        if model_dir.is_file():
            weights_path = model_dir
            config_path = model_dir.parent / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config at {config_path}")
        else:
            config_path = model_dir / "config.json"
            weights_path = model_dir / "pytorch_model.bin"
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Missing model weights at {weights_path}. "
                    "Pass a checkpoint directory (containing config.json and pytorch_model.bin) "
                    "or the pytorch_model.bin file itself."
                )
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config at {config_path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        fields = {k: payload[k] for k in ConditionalMidiConfig.__dataclass_fields__ if k in payload}
        config = ConditionalMidiConfig(**fields)
        model = cls(config)
        state = torch.load(weights_path, map_location=map_location)
        model, (missing, unexpected) = cls._load_state_into_model(model, state, payload, config.vocab_size)
        if missing:
            print(f"from_pretrained: missing keys (randomly initialised): {missing}")
        if unexpected:
            print(f"from_pretrained: unexpected keys (ignored): {unexpected}")
        return model, payload

    @classmethod
    def resume_from_checkpoint(
        cls,
        ckpt_dir: str | Path,
        config: ConditionalMidiConfig,
        map_location: str | torch.device = "cpu",
    ) -> tuple["ConditionalMidiTransformer", dict, tuple[list[str], list[str]]]:
        """Build a model for `config` and load weights from a (possibly larger-vocab) checkpoint."""
        ckpt_dir = Path(ckpt_dir)
        checkpoint_config = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
        model = cls(config)
        state = torch.load(ckpt_dir / "pytorch_model.bin", map_location=map_location)
        _, load_info = cls._load_state_into_model(model, state, checkpoint_config, config.vocab_size)
        return model, checkpoint_config, load_info

    @classmethod
    def _load_state_into_model(
        cls,
        model: "ConditionalMidiTransformer",
        state: dict[str, torch.Tensor],
        checkpoint_config: dict,
        target_vocab_size: int,
    ) -> tuple["ConditionalMidiTransformer", tuple[list[str], list[str]]]:
        ckpt_vocab = checkpoint_config.get("vocab_size", target_vocab_size)
        if ckpt_vocab != target_vocab_size:
            state = adapt_checkpoint_state_dict(
                state,
                ckpt_vocab_size=ckpt_vocab,
                target_vocab_size=target_vocab_size,
            )
        # Allow partial load when condition_embedding size changed (e.g. pretrain -> finetune
        # with expanded composer/genre vocab).  Missing keys are randomly initialised.
        missing, unexpected = model.load_state_dict(state, strict=False)
        return model, (missing, unexpected)


def _sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / max(temperature, 1e-6)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)
