"""The ten multi-task heads (architecture: Multi-Task Heads).

Every head is a branching MLP over the *shared* backbone, so predictions come out
of the same forward pass that produced the alignment — the marginal cost of a head
is one small MLP, not a second model. Heads split by what they attach to:

- **Per-read** (from the pooled embedding): haplotype phase, HLA type, ancestry,
  somatic status, PGx star allele.
- **Per-graph-node** (from the refined node states): variant genotype + GQ, SV
  type + breakpoints, copy number.
- **Per-read-base** (from the read states): recalibrated base quality (BQSR),
  CpG methylation.

Heads are opt-in via :class:`~graphmambaformer.config.MultiTaskConfig` because each
needs its own labels; :class:`MultiTaskHeads` builds only those enabled and reports
which outputs it produced so the loss can match them up.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import MultiTaskConfig


def _mlp(d_in: int, d_hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    """The standard branching MLP shared by every head."""
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, d_out),
    )


class MultiTaskHeads(nn.Module):
    """Container for the enabled task heads.

    ``forward`` returns a flat dict of raw logits/regressions keyed by head name;
    activations are left to the loss so training stays numerically stable
    (``cross_entropy`` and ``binary_cross_entropy_with_logits`` both want logits).
    """

    #: Which representation each head consumes.
    SCOPES: dict[str, str] = {
        "variant_calling": "node",
        "sv_genotyping": "node",
        "copy_number": "node",
        "haplotype": "read",
        "hla_typing": "read",
        "ancestry": "read",
        "somatic": "read",
        "pgx": "read",
        "bqsr": "base",
        "methylation": "base",
    }

    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()
        self.cfg = cfg
        d, h, p = cfg.d_model, cfg.d_hidden, cfg.dropout
        self.heads = nn.ModuleDict()

        # Per-node heads.
        if cfg.variant_calling:
            # Genotype class plus a genotype-quality regression per bubble node.
            self.heads["variant_calling"] = _mlp(d, h, cfg.num_genotypes + 1, p)
        if cfg.sv_genotyping:
            # SV type plus two breakpoint offsets.
            self.heads["sv_genotyping"] = _mlp(d, h, cfg.num_sv_types + 2, p)
        if cfg.copy_number:
            # Discrete CN state plus a continuous CN estimate.
            self.heads["copy_number"] = _mlp(d, h, cfg.num_cn_states + 1, p)

        # Per-read heads.
        if cfg.haplotype:
            self.heads["haplotype"] = _mlp(d, h, 2, p)  # phase 0 / 1
        if cfg.hla_typing:
            self.heads["hla_typing"] = _mlp(d, h, cfg.num_hla_alleles, p)
        if cfg.ancestry:
            # Global 5-population softmax; local painting reuses the node scope.
            self.heads["ancestry"] = _mlp(d, h, cfg.num_populations, p)
            self.heads["ancestry_local"] = _mlp(d, h, cfg.num_populations, p)
        if cfg.somatic:
            self.heads["somatic"] = _mlp(d, h, cfg.num_somatic_classes, p)
        if cfg.pgx:
            self.heads["pgx"] = _mlp(d, h, cfg.num_pgx_alleles, p)

        # Per-base heads.
        if cfg.bqsr:
            self.heads["bqsr"] = _mlp(d, h, cfg.num_quality_bins, p)
        if cfg.methylation:
            self.heads["methylation"] = _mlp(d, h, 2, p)  # unmethylated / methylated

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(self.heads.keys())

    def forward(
        self,
        pooled: torch.Tensor,
        read_states: torch.Tensor | None = None,
        node_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run every enabled head.

        Args:
            pooled: ``(B, D)`` per-read embedding.
            read_states: ``(B, L, D)`` per-base states (needed by BQSR/methylation).
            node_states: ``(B, N, D)`` per-node states (needed by the node heads).
        """
        out: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            scope = self.SCOPES.get(name.removesuffix("_local"), "read")
            if name == "ancestry_local":
                scope = "node"

            if scope == "node":
                if node_states is None:
                    continue
                out[name] = head(node_states)
            elif scope == "base":
                if read_states is None:
                    continue
                out[name] = head(read_states)
            else:
                out[name] = head(pooled)
        return out
