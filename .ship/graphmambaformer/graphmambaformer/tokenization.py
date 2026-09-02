"""DNA k-mer tokenization utilities shared by the read and graph encoders.

A k-mer tokenizer maps a nucleotide string to a sequence of integer token ids.
Reserved ids: ``PAD = 0`` and ``UNK = 1`` (used for any k-mer containing a
non-ACGT character such as ``N``). Real k-mer ids start at ``offset = 2``.
"""

from __future__ import annotations

from typing import Sequence

import torch

_BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


class KmerTokenizer:
    PAD_ID = 0
    UNK_ID = 1
    OFFSET = 2

    def __init__(self, k: int = 3, stride: int = 1):
        if k < 1:
            raise ValueError("k must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.k = k
        self.stride = stride

    @property
    def vocab_size(self) -> int:
        return 4**self.k + self.OFFSET

    def encode(self, seq: str) -> list[int]:
        """Encode a nucleotide string into a list of k-mer token ids."""
        seq = seq.upper()
        ids: list[int] = []
        for i in range(0, len(seq) - self.k + 1, self.stride):
            kmer = seq[i : i + self.k]
            idx = 0
            ok = True
            for ch in kmer:
                base = _BASE2IDX.get(ch)
                if base is None:
                    ok = False
                    break
                idx = idx * 4 + base
            ids.append(self.OFFSET + idx if ok else self.UNK_ID)
        return ids

    def encode_quality(self, quals: Sequence[int]) -> list[float]:
        """Aggregate per-base Phred qualities into per-k-mer qualities (mean)."""
        out: list[float] = []
        for i in range(0, len(quals) - self.k + 1, self.stride):
            window = quals[i : i + self.k]
            out.append(sum(window) / len(window))
        return out

    def batch_encode(
        self,
        seqs: Sequence[str],
        quals: Sequence[Sequence[int]] | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode + right-pad a batch of reads.

        Returns a dict with:
          - ``token_ids``: LongTensor ``(B, L)``
          - ``mask``: BoolTensor ``(B, L)`` (True = real token)
          - ``qualities``: FloatTensor ``(B, L)`` (zeros if ``quals`` is None)
        """
        token_lists = [self.encode(s) for s in seqs]
        max_len = max((len(t) for t in token_lists), default=0)
        max_len = max(max_len, 1)
        B = len(seqs)

        token_ids = torch.full((B, max_len), self.PAD_ID, dtype=torch.long)
        mask = torch.zeros((B, max_len), dtype=torch.bool)
        qual_tensor = torch.zeros((B, max_len), dtype=torch.float32)

        qual_lists = None
        if quals is not None:
            qual_lists = [self.encode_quality(q) for q in quals]

        for b, toks in enumerate(token_lists):
            n = len(toks)
            token_ids[b, :n] = torch.tensor(toks, dtype=torch.long)
            mask[b, :n] = True
            if qual_lists is not None:
                qk = qual_lists[b]
                qual_tensor[b, : len(qk)] = torch.tensor(qk, dtype=torch.float32)

        out = {"token_ids": token_ids, "mask": mask, "qualities": qual_tensor}
        if device is not None:
            out = {key: value.to(device) for key, value in out.items()}
        return out
