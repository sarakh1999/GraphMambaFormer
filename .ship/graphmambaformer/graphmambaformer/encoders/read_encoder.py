"""Modality-Aware Read Encoder (Figure 1A, left).

k-mer tokenization + base-quality embedding + sinusoidal positional encoding +
modality-conditioning token -> ``d_model`` sequence of read embeddings.

Development focus: long reads (PacBio HiFi, ONT), but the modality embedding
table covers all registered modalities so the same encoder generalizes later.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ReadEncoderConfig, modality_id
from ..layers.common import SinusoidalPositionalEncoding
from ..tokenization import KmerTokenizer


class ModalityAwareReadEncoder(nn.Module):
    def __init__(self, cfg: ReadEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = KmerTokenizer(k=cfg.kmer_size, stride=cfg.kmer_stride)

        self.token_emb = nn.Embedding(
            self.tokenizer.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx
        )
        # Base-quality embedding over discrete Phred bins.
        self.quality_emb = nn.Embedding(cfg.max_quality, cfg.d_model)
        # Modality-conditioning token (also prepended to the sequence).
        self.modality_emb = nn.Embedding(cfg.num_modalities, cfg.d_model)

        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.input_norm = nn.LayerNorm(cfg.d_model)

    # ---- convenience: go straight from raw reads to embeddings -------------- #
    def encode_reads(
        self,
        seqs: list[str],
        modality: str | int | list[str | int],
        quals: list[list[int]] | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenizer.batch_encode(seqs, quals=quals, device=device)
        return self.forward(
            token_ids=batch["token_ids"],
            qualities=batch["qualities"],
            modality=modality,
            mask=batch["mask"],
        )

    def _modality_tensor(
        self, modality: str | int | list[str | int], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if isinstance(modality, (list, tuple)):
            ids = [modality_id(m) if isinstance(m, str) else int(m) for m in modality]
        else:
            single = modality_id(modality) if isinstance(modality, str) else int(modality)
            ids = [single] * batch_size
        return torch.tensor(ids, dtype=torch.long, device=device)

    def forward(
        self,
        token_ids: torch.Tensor,
        modality: str | int | list[str | int],
        qualities: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(hidden, mask)`` with hidden of shape ``(B, L', d_model)``.

        ``L'`` includes the prepended modality token when
        ``cfg.prepend_modality_token`` is True.
        """
        B, L = token_ids.shape
        device = token_ids.device

        h = self.token_emb(token_ids)

        if qualities is not None:
            q_bins = qualities.round().long().clamp_(0, self.cfg.max_quality - 1)
            h = h + self.quality_emb(q_bins)

        if self.cfg.use_positional_encoding:
            h = self.pos_enc(h)

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        mod_ids = self._modality_tensor(modality, B, device)
        if self.cfg.prepend_modality_token:
            mod_tok = self.modality_emb(mod_ids).unsqueeze(1)  # (B, 1, d_model)
            h = torch.cat([mod_tok, h], dim=1)
            mask = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=device), mask], dim=1
            )
        else:
            # Add modality embedding as a global bias to every position instead.
            h = h + self.modality_emb(mod_ids).unsqueeze(1)

        h = self.input_norm(h)
        h = self.dropout(h)
        return h, mask
