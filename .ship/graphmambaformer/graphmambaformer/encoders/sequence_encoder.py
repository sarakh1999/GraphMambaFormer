"""GraphMamba SequenceEncoder (architecture: Input Encoding).

Four complementary views of every base are concatenated and projected to
``d_model``::

    Base(5 -> 64)  +  Kmer(k=3 -> 64)  +  Qual(42 -> 32)  +  PosEncode(sin/cos -> 96)
        -> Linear(256 -> D) -> LayerNorm -> Dropout(0.1)

Each view carries information the others cannot. The per-base embedding keeps
single-nucleotide identity that a k-mer embedding blurs; the k-mer embedding
supplies local sequence context (so homopolymers and short repeats are
distinguishable); the quality embedding tells the model which bases to distrust,
which is what makes it possible to learn platform-specific error profiles; and
the sinusoidal encoding is computed on the fly so a 100 kb ONT read needs no
position table.

Unlike :class:`~graphmambaformer.encoders.read_encoder.ModalityAwareReadEncoder`,
which tokenizes into k-mer space and therefore shortens the sequence by ``k - 1``,
this encoder stays in **base space**: output position ``i`` is read base ``i``.
Stage 4 needs that, because it indexes the hidden states by anchor read position.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..alignment.seeding import N_CODE, encode_bases
from ..config import SequenceEncoderConfig, modality_id
from ..layers.common import SinusoidalPositionalEncoding

# Base codes are shared with the seeding stage: 0 pad, 1-4 ACGT, 5 N.
PAD_CODE = 0
NUM_BASE_SYMBOLS = 6  # pad + A C G T N


class SequenceEncoder(nn.Module):
    """Encode reads (base codes + qualities) into ``(B, L, d_model)`` states."""

    def __init__(self, cfg: SequenceEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.base_emb = nn.Embedding(NUM_BASE_SYMBOLS, cfg.d_base, padding_idx=PAD_CODE)
        # k-mer vocabulary: 4**k real k-mers plus a shared slot for any window
        # containing a pad or an N.
        self.kmer_vocab = 4**cfg.kmer_size + 1
        self.kmer_unk = self.kmer_vocab - 1
        self.kmer_emb = nn.Embedding(self.kmer_vocab, cfg.d_kmer)
        self.qual_emb = nn.Embedding(cfg.num_quality_bins, cfg.d_qual)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_pos)

        self.proj = nn.Linear(cfg.d_concat, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.modality_emb = (
            nn.Embedding(cfg.num_modalities, cfg.d_model)
            if cfg.prepend_modality_token
            else None
        )

    # ---- host-side helpers -------------------------------------------------- #
    @staticmethod
    def encode_batch(
        reads: list[str],
        qualities: list[list[int]] | None = None,
        max_len: int | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Pad a batch of read strings into ``base_codes`` / ``qualities`` / ``mask``.

        Base codes match :func:`~graphmambaformer.alignment.seeding.encode_bases`,
        so the tensors the model sees and the arrays the seeding stage indexes are
        the same alphabet.
        """
        encoded = [encode_bases(r) for r in reads]
        if max_len is not None:
            encoded = [c[:max_len] for c in encoded]
        width = max((len(c) for c in encoded), default=1) or 1

        B = len(reads)
        codes = np.zeros((B, width), dtype=np.int64)
        quals = np.zeros((B, width), dtype=np.int64)
        mask = np.zeros((B, width), dtype=bool)
        for b, seq in enumerate(encoded):
            n = len(seq)
            codes[b, :n] = seq
            mask[b, :n] = True
            if qualities is not None:
                q = np.asarray(qualities[b][:n], dtype=np.int64)
                quals[b, : len(q)] = q

        out = {
            "base_codes": torch.as_tensor(codes),
            "qualities": torch.as_tensor(quals),
            "mask": torch.as_tensor(mask),
        }
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out

    # ---- k-mer view --------------------------------------------------------- #
    def _kmer_ids(self, base_codes: torch.Tensor) -> torch.Tensor:
        """Centred k-mer id per position, computed on device.

        The window is centred so position ``i``'s k-mer describes the context
        *around* base ``i`` rather than the context that follows it; edges are
        replicate-padded. Any window touching a pad or an ``N`` maps to the shared
        unknown slot.
        """
        k = self.cfg.kmer_size
        B, L = base_codes.shape
        left = (k - 1) // 2

        digits = base_codes - 1  # A C G T -> 0..3; pad -> -1; N -> 4
        invalid = (digits < 0) | (digits > 3)
        digits = digits.clamp(min=0)

        padded_digits = torch.nn.functional.pad(digits, (left, k - 1 - left), mode="replicate" if L > 1 else "constant")
        padded_invalid = torch.nn.functional.pad(invalid.to(digits.dtype), (left, k - 1 - left), value=1)

        ids = torch.zeros((B, L), dtype=torch.long, device=base_codes.device)
        bad = torch.zeros((B, L), dtype=torch.bool, device=base_codes.device)
        for offset in range(k):
            ids = ids * 4 + padded_digits[:, offset : offset + L]
            bad |= padded_invalid[:, offset : offset + L].bool()
        return torch.where(bad, torch.full_like(ids, self.kmer_unk), ids)

    # ---- forward ------------------------------------------------------------ #
    def forward(
        self,
        base_codes: torch.Tensor,
        qualities: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        modality: str | int | list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of reads.

        Args:
            base_codes: ``(B, L)`` base codes (0 pad, 1-4 ACGT, 5 N).
            qualities: ``(B, L)`` Phred scores, clamped into the embedding range.
            mask: ``(B, L)`` bool, True for real bases (inferred from pads if None).
            modality: modality name/id (or one per read) for the conditioning token.

        Returns ``(hidden, mask)`` of shape ``(B, L, d_model)`` / ``(B, L)``, or
        ``(B, L + 1, ...)`` when a modality token is prepended.
        """
        if mask is None:
            mask = base_codes != PAD_CODE

        parts = [
            self.base_emb(base_codes),
            self.kmer_emb(self._kmer_ids(base_codes)),
        ]
        if qualities is None:
            qualities = torch.zeros_like(base_codes)
        parts.append(
            self.qual_emb(qualities.clamp(0, self.cfg.num_quality_bins - 1).long())
        )

        positional = self.pos_enc(
            torch.zeros(
                base_codes.shape[0],
                base_codes.shape[1],
                self.cfg.d_pos,
                device=base_codes.device,
                dtype=parts[0].dtype,
            )
        )
        parts.append(positional)

        hidden = self.dropout(self.norm(self.proj(torch.cat(parts, dim=-1))))
        # Padded positions must not leak into the SSM scan or the attention.
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)

        if self.modality_emb is not None:
            token = self._modality_token(modality, base_codes.shape[0], hidden.device)
            hidden = torch.cat([token.unsqueeze(1), hidden], dim=1)
            mask = torch.cat(
                [torch.ones_like(mask[:, :1]), mask], dim=1
            )
        return hidden, mask

    def _modality_token(
        self, modality: str | int | list | None, batch: int, device: torch.device
    ) -> torch.Tensor:
        if modality is None:
            ids = torch.zeros(batch, dtype=torch.long, device=device)
        elif isinstance(modality, list):
            ids = torch.tensor(
                [m if isinstance(m, int) else modality_id(m) for m in modality],
                dtype=torch.long,
                device=device,
            )
        else:
            value = modality if isinstance(modality, int) else modality_id(modality)
            ids = torch.full((batch,), value, dtype=torch.long, device=device)
        return self.modality_emb(ids)
