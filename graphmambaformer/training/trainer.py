"""Training and validation loop for the GraphMamba alignment model.

Runs on whatever the host has -- CUDA (any generation), ROCm, XPU, MPS or CPU --
by taking every device decision from :class:`AccelContext` rather than testing
for CUDA inline. The fp16 path carries a ``GradScaler``; bf16 does not need one.

Two choices are deliberate:

* Validation drives early stopping on a *metric*, not on validation loss. The
  objective is a weighted sum of seven terms whose balance shifts as the Kendall
  weights learn, so its absolute value is not comparable across epochs.
* Every step is instrumented (see :mod:`.probes`). The history is kept so the
  plots can show what the model did, not merely what it scored.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from ..accel import (
    AccelContext,
    Prefetcher,
    configure_torch_threads,
    default_worker_count,
)
from ..config import AccelConfig, LossConfig
from ..device import resolve_device_ids, unwrap_model, wrap_data_parallel
from ..distributed import DistContext, maybe_init_distributed
from ..losses import GraphMambaLoss
from ..progress import progress
from .metrics import (
    ValidationMetrics,
    anchor_metrics,
    chain_accuracy,
    locus_accuracy,
    mapq_calibration,
    mapq_mae,
)
from .probes import BehaviorProbe, StepReport
from .targets import Supervision, TargetBuilder

__all__ = ["TrainConfig", "TrainHistory", "Trainer"]


@dataclass
class TrainConfig:
    """Knobs for a training run. Defaults are sized for a quick CPU smoke run."""

    epochs: int = 8
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    #: Accumulate gradients over this many micro-batches before each optimizer
    #: step, so the *effective* batch size is ``batch_size * grad_accum`` without
    #: the memory of a larger batch. 1 = step every batch (original behaviour).
    #: The loss is averaged over the window, the LR schedule counts optimizer
    #: steps (not micro-batches), and the tail of an epoch is flushed even when
    #: it does not fill a full window so no batch is dropped.
    grad_accum: int = 1
    warmup_frac: float = 0.1
    #: Stop when the monitored metric has not improved for this many epochs.
    patience: int = 4
    #: Early-stopping metric. Falls back to -val_loss if it is unmeasurable on
    #: the data at hand (see ValidationMetrics.monitored).
    monitor: str = "locus_accuracy"
    log_every: int = 1
    #: How often (in steps) to flush ``history.json`` to disk during an epoch.
    #: The whole (growing) step log is re-serialised each flush, so doing it
    #: every step is O(steps^2) disk work on the same thread that drives the
    #: GPU -- it stalls the accelerator for progressively longer as a run goes
    #: on. Flushing periodically (plus always at every epoch boundary) keeps the
    #: crash-safe log without starving the device. Set to 1 for the old cadence.
    flush_every: int = 50
    #: Run the full behaviour probe (per-parameter grad norms, activation stats,
    #: router/head spread, label balance) every N steps. Each of those forces
    #: GPU->CPU synchronisations that stall the accelerator; sampling them keeps
    #: the diagnostic curves informative at a fraction of the cost. The loss and
    #: its terms are still recorded every step. Set to 1 for the old per-step
    #: instrumentation.
    probe_every: int = 10
    seed: int = 0
    out_dir: str = "data/training_runs/latest"
    #: If set, write ``checkpoint.pt`` (best weights) under ``out_dir``.
    save_checkpoint: bool = True
    #: If set, also write ``checkpoints/epoch_XX.pt`` after every epoch, so no
    #: intermediate state is ever lost (the user can resume/inspect any epoch).
    save_every_epoch: bool = True
    #: If >0, additionally write ``checkpoints/step_XXXXXX.pt`` every N optimizer
    #: steps *within* an epoch, so long runs are recoverable at a fine grain
    #: (independent of the epoch boundary). 0 disables step checkpoints.
    save_every_steps: int = 0
    #: If >0, redraw the figures into ``out_dir/plots`` every N optimizer steps
    #: *within* an epoch, so a long-epoch run has current loss/behaviour curves
    #: long before the first epoch boundary. 0 = per-epoch only (unchanged).
    plot_every_steps: int = 0
    #: If >0, run a validation pass every N optimizer steps *within* an epoch and
    #: append the metrics to the history (so validation curves fill in mid-epoch).
    #: 0 = validate only at epoch boundaries (unchanged).
    validate_every_steps: int = 0
    #: Cap intra-epoch validation to this many val batches (0 = the full val set).
    #: Intra-epoch validation also runs the end-to-end aligner, which is costly on
    #: a large val set, so a small cap keeps the mid-epoch check cheap.
    intra_val_max_batches: int = 0
    #: Cap the *epoch-boundary* validation to this many val batches (0 = full
    #: pass). The end-of-epoch validate() runs the whole aligner over every val
    #: batch, which can dominate epoch time on a large held-out set; capping it
    #: keeps a solid estimate while reclaiming hours per epoch. Intra-epoch
    #: validation has its own (usually smaller) cap above.
    epoch_val_max_batches: int = 0
    #: If set, keep ``last.pt`` pointing at the most recent epoch's weights.
    save_last: bool = True
    #: If set, flush ``history.json`` to disk after every epoch (crash-safe log).
    checkpoint_history: bool = True
    #: If set, (re)draw the training figures into ``out_dir/plots`` after every
    #: epoch, so the loss curve (and the rest) is available and kept current
    #: during a long run instead of only at the end. A no-op when matplotlib is
    #: missing, and wrapped so plotting can never abort training.
    plot_every_epoch: bool = True
    #: Multi-GPU device list: ``"auto"`` (all visible CUDA/XPU when count>1),
    #: ``"all"``, ``"0,1"``, or ``"none"`` for single-device.
    devices: str = "auto"


@dataclass
class TrainHistory:
    """Everything recorded during a run, ready for plotting or JSON."""

    steps: list[StepReport] = field(default_factory=list)
    epochs: list[dict] = field(default_factory=list)
    validations: list[ValidationMetrics] = field(default_factory=list)
    device_summary: str = ""
    config: dict = field(default_factory=dict)

    def to_json(self, path: str) -> str:
        payload = {
            "device": self.device_summary,
            "config": self.config,
            "steps": [asdict(s) for s in self.steps],
            "epochs": self.epochs,
            "validations": [asdict(v) for v in self.validations],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        return path


class Trainer:
    """Trains the core model on supervision built from ground-truth reads."""

    def __init__(self, model, pipeline, cfg: Optional[TrainConfig] = None,
                 loss_cfg: Optional[LossConfig] = None,
                 accel: Optional[AccelContext] = None,
                 verbose: bool = True,
                 dist_ctx: Optional[DistContext] = None):
        self.cfg = cfg or TrainConfig()
        self.pipeline = pipeline

        # Distributed data-parallel context (one process per GPU under
        # ``torchrun``). When not launched distributed this is a disabled
        # context whose collectives are all no-ops, so the single-GPU / CPU path
        # is unchanged. Only the main rank logs, to avoid N-way duplicate output.
        self.dist = dist_ctx if dist_ctx is not None else maybe_init_distributed()
        self.verbose = verbose and self.dist.is_main

        self.accel = accel or AccelContext(AccelConfig())
        self.device = self.accel.caps.device

        # Size the host thread pools once, so the CPU-bound stages that feed the
        # GPU (seeding/chaining/supervision) use every core instead of one.
        if getattr(self.accel.cfg, "set_threads", True):
            info = configure_torch_threads(
                self.device.type, workers=self.accel.cfg.num_workers
            )
            if self.verbose and not info.get("skipped"):
                print(f"host threads: torch={info.get('num_threads')} "
                      f"(cores={info.get('requested_cores')})")
        # How far ahead to build batches, and how many CPU workers feed the GPU.
        self._feed_workers = default_worker_count(self.accel.cfg.num_workers)
        self._prefetch = int(getattr(self.accel.cfg, "prefetch", 0))

        if self.dist.enabled:
            # Under torchrun each rank owns exactly one GPU and gradients are
            # synchronised via all-reduce (see ``train_epoch``); DataParallel is
            # never nested inside the process group. The primary device was
            # already pinned to LOCAL_RANK by ``maybe_init_distributed``.
            if self.device.type == "cuda":
                idx = self.device.index if self.device.index is not None else self.dist.local_rank
                self.device_ids = [int(idx)]
            else:
                self.device_ids = []
        else:
            self.device_ids = resolve_device_ids(self.cfg.devices, primary=self.device)
            if len(self.device_ids) > 1:
                # nn.DataParallel cannot gather this model's custom dataclass
                # output and only wraps the backbone forward (the scoring heads
                # run on the unwrapped module), so it is unsupported here. Fall
                # back to the single primary device instead of crashing, and
                # point the user at the correct multi-GPU launch (torchrun).
                if self.verbose:
                    print(
                        f"warning: --devices {self.cfg.devices!r} requests "
                        f"{len(self.device_ids)} GPUs via nn.DataParallel, which is "
                        "not supported for GraphMamba (custom model output). "
                        "Using a single GPU. For multi-GPU, launch with "
                        "`torchrun --standalone --nproc_per_node=<N> scripts/train.py ...`."
                    )
                self.device_ids = self.device_ids[:1]

        if self.device_ids and self.device.type in ("cuda", "xpu"):
            # Primary compute device is the first id in the (now single) list.
            self.device = torch.device(self.device.type, int(self.device_ids[0]))
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)

        model = model.to(self.device)
        self.model = wrap_data_parallel(model, self.device_ids)
        # Keep the true core module for the scoring heads / checkpoints; it must
        # stay un-compiled so attribute access and state_dict keys are stable.
        self.raw_model = unwrap_model(self.model)
        # torch.compile fuses the encoder/mamba/mapping forward. Skipped on MPS
        # and under DataParallel (compiled replicas are fragile); opt-in via
        # AccelConfig.compile. The heads still call the eager ``raw_model``.
        if getattr(self.accel.cfg, "compile", False) and len(self.device_ids) <= 1:
            compiled = self.accel.compile(self.model)
            if compiled is not self.model and self.verbose:
                print(f"torch.compile: on (mode={self.accel.cfg.compile_mode})")
            self.model = compiled

        self.criterion = GraphMambaLoss(loss_cfg or LossConfig()).to(self.device)
        # Supervision must be encoded with the same read-length cap the pipeline
        # forward uses, else long reads (HiFi/ONT) blow up the O(L^2) towers.
        self.builder = TargetBuilder(
            pipeline, model=self.raw_model,
            max_read_len=getattr(getattr(pipeline, "cfg", None), "max_read_len", None),
        )
        self.probe = BehaviorProbe(self.raw_model)

        # Kendall log-variances are parameters too, so they must be optimized.
        params = list(self.raw_model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        # Exact parameter list whose gradients are averaged across ranks each
        # optimizer step (the same set the optimizer updates: backbone + heads +
        # the criterion's learnable weights).
        self._optim_params = params
        self.scaler = self.accel.grad_scaler()
        summary = self.accel.summary()
        if self.dist.enabled:
            summary = (f"{summary} | ddp(world_size={self.dist.world_size}, "
                       f"rank={self.dist.rank}, backend={self.dist.backend})")
        self.history = TrainHistory(
            device_summary=summary, config=asdict(self.cfg)
        )
        self._step = 0
        if self.verbose and self.dist.enabled:
            print(f"multi-GPU DDP: world_size={self.dist.world_size} "
                  f"(this rank {self.dist.rank} on {self.device}); gradients "
                  "all-reduced per optimizer step, rank 0 writes checkpoints")

    # ---- one forward pass -------------------------------------------------- #
    def _forward(self, sup: Supervision, reference):
        """Model forward plus both scoring heads, on the resolved device."""
        # DataParallel parallelizes ``forward`` across GPUs; scoring heads stay
        # on the primary device via ``raw_model`` (gathered outputs live there).
        outputs = self.model(
            sup.base_codes,
            mask=sup.mask,
            graph=reference.graph,
            qualities=sup.qualities,
            modality=sup.modality,
        )
        seed_scores = self.raw_model.score_seeds(
            outputs,
            seed_features=sup.seed_features,
            anchor_read_pos=sup.anchor_read_pos,
            anchor_node=sup.anchor_node,
            anchor_mask=sup.anchor_mask,
            edge_index=sup.seed_edge_index,
            edge_features=sup.seed_edge_features,
            edge_mask=sup.seed_edge_mask,
            gnn_active=sup.seed_gnn_active,
        )
        b, n_chain, n_members = sup.member_states_shape
        # Member states are the backbone's ``read_hidden`` at each chain member
        # anchor's read position -- the SAME representation the inference
        # re-ranker pools (NeuralScorer.score_chains). Training previously fed
        # zeros here, which left the chain head's member-pooling branch untrained
        # AND created a train/serve skew (real states at inference then flowed
        # through that untrained branch, corrupting the chain scores). Gather the
        # real states from the supervision's member positions so train == serve.
        read_hidden = outputs.read_hidden
        d_model = read_hidden.shape[-1]
        length = read_hidden.shape[1]
        if sup.chain_member_pos is not None and n_members > 0:
            member_pos = sup.chain_member_pos.to(read_hidden.device)
            member_mask = sup.chain_member_mask.to(read_hidden.device)
            flat = member_pos.clamp(0, max(length - 1, 0)).reshape(b, -1)
            member_states = read_hidden.gather(
                1, flat.unsqueeze(-1).expand(-1, -1, d_model)
            ).reshape(b, n_chain, n_members, d_model)
            member_states = member_states * member_mask.unsqueeze(-1).to(member_states.dtype)
        else:
            # Back-compat (e.g. a supervision built before this field existed):
            # fall back to the previous zero-member behaviour.
            member_states = read_hidden.new_zeros((b, n_chain, n_members, d_model))
            member_mask = torch.ones(
                b, n_chain, n_members, dtype=torch.bool, device=read_hidden.device
            )
        chain_scores = self.raw_model.score_chains(
            chain_features=sup.chain_feats,
            member_states=member_states,
            member_mask=member_mask,
            chain_mask=sup.chain_mask,
        )
        return outputs, seed_scores, chain_scores

    def _feed(self, batches: Sequence[tuple]):
        """Yield ``(supervision_on_device, reference)`` with look-ahead.

        The classical stages that build supervision run on the host every step,
        so building the *next* batch on a background thread while the GPU is busy
        with the current one is what keeps the accelerator from idling. The heavy
        seeding/chaining runs in the worker; only the (cheap, ordered) device
        transfer happens on the main thread, which keeps CUDA calls serialized.
        """
        def build(item):
            reads, reference = item
            return self.builder.build(reads, reference)

        feed = Prefetcher(
            build, list(batches), depth=self._prefetch, workers=self._feed_workers
        )
        for sup, item in feed:
            yield sup.to(self.device), item[1]

    def _lr_at(self, step: int, total: int) -> float:
        """Linear warmup then cosine decay."""
        warmup = max(1, int(total * self.cfg.warmup_frac))
        if step < warmup:
            return self.cfg.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return self.cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    # ---- epochs ------------------------------------------------------------ #
    def train_epoch(self, batches: Sequence[tuple], epoch: int,
                    total_steps: int,
                    val_batches: Optional[Sequence[tuple]] = None) -> dict:
        """``batches`` is a sequence of ``(reads, reference)`` pairs.

        Carrying the reference per batch rather than once for the whole run is
        what lets a dataset with several references contribute all of its reads
        instead of only those on the first one.

        ``val_batches`` is only used for intra-epoch validation
        (``validate_every_steps``); the per-epoch validation still runs from
        :meth:`fit`. When the intra-epoch knobs are 0 it is ignored entirely.
        """
        self.model.train()
        losses: list[float] = []
        started = time.time()

        accum = max(1, int(getattr(self.cfg, "grad_accum", 1)))
        n_batches = len(batches)

        batch_bar = progress(
            self._feed(batches),
            total=n_batches,
            desc=f"train epoch {epoch:02d}",
            unit="batch",
            disable=not self.verbose,
            leave=False,
        )
        # Accumulate gradients over ``accum`` micro-batches, then step once. The
        # window's losses are averaged (scaled by 1/accum on each backward), so
        # the update matches a single batch of size ``batch_size * accum``. LR,
        # probes, checkpoints and the step log all key off the optimizer step
        # (``self._step``), not the micro-batch, so their cadence is unchanged.
        self.optimizer.zero_grad(set_to_none=True)
        window_losses: list[float] = []
        for micro_i, (sup, reference) in enumerate(batch_bar):
            is_step = ((micro_i + 1) % accum == 0) or (micro_i + 1 == n_batches)

            lr = self._lr_at(self._step, total_steps)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            # Sample the sync-heavy behaviour probe every ``probe_every`` steps.
            # The flag is read by the forward hooks, so it must be set before the
            # forward pass runs; only meaningful on a step boundary (grads exist).
            do_probe = is_step and (self._step % max(1, self.cfg.probe_every) == 0)
            self.probe.active = do_probe

            with self.accel.precision():
                outputs, seed_scores, chain_scores = self._forward(sup, reference)
                loss = self.criterion(
                    outputs, sup.targets,
                    seed_scores=seed_scores, chain_scores=chain_scores,
                )

            # Scale so the summed gradient over the window equals the mean loss.
            scaled = loss.total / accum
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            losses.append(float(loss.total.detach()))
            window_losses.append(float(loss.total.detach()))

            if not is_step:
                # Mid-window: gradient is accumulating, no step/report yet.
                continue

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Data-parallel gradient sync: average this rank's accumulated
            # gradient with every other rank's, once per optimizer step. Done
            # after unscaling (so the average is in real, unscaled units) and
            # before clipping/reporting (so clip_grad_norm_ and the probe see the
            # synchronized gradient). A no-op in a single-process run, so the
            # single-GPU gradient is bit-for-bit unchanged.
            self.dist.average_gradients(self._optim_params)

            # Report gradients after unscaling but before clipping, so the norms
            # describe what the model produced rather than what was allowed. On
            # non-sampled steps a cheap report avoids the probe's GPU->CPU syncs
            # (and the label-balance reduction) entirely.
            if do_probe:
                report = self.probe.report(
                    self._step, epoch, loss, outputs, split="train",
                    seed_scores=seed_scores, chain_scores=chain_scores, lr=lr,
                    label_balance=TargetBuilder.label_balance(sup),
                )
            else:
                report = self.probe.light_report(
                    self._step, epoch, loss, split="train", lr=lr
                )
            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # Log the window-averaged loss so the curve is comparable whatever
            # the accumulation factor. Under DDP average it across ranks too so
            # the logged curve reflects the global batch, not just this rank's.
            report.total = float(np.mean(window_losses)) if window_losses else report.total
            report.total = self.dist.reduce_mean(report.total)
            window_losses = []

            self.history.steps.append(report)
            if hasattr(batch_bar, "set_postfix"):
                batch_bar.set_postfix(loss=f"{report.total:.4f}", refresh=False)
            if self.verbose and self._step % self.cfg.log_every == 0:
                print("  " + report.one_line())
            # Flush the step log periodically so a crash never loses much work,
            # without paying the O(steps^2) re-serialisation cost every step
            # (which stalls the GPU). ``_persist_epoch`` also flushes at every
            # epoch boundary, so the on-disk log is never more than one flush
            # interval behind.
            if self.cfg.checkpoint_history and self.dist.is_main and (
                self._step % max(1, self.cfg.flush_every) == 0
            ):
                self.history.to_json(os.path.join(self.cfg.out_dir, "history.json"))
            self._step += 1
            # Fine-grained step checkpoints for long runs: independent of the
            # epoch boundary so nothing is lost between epochs.
            if self.cfg.save_every_steps and (
                self._step % self.cfg.save_every_steps == 0
            ):
                self._persist_step(epoch=epoch)

            # Intra-epoch validation + plotting. Both run here — at a clean
            # optimizer-step boundary, after step()+zero_grad() — so no
            # half-accumulated gradients are pending and the model can safely
            # switch to eval and back. Guarded by knobs that default to 0, so
            # when they are off this whole block is skipped and behaviour is
            # identical to the per-epoch-only path.
            if (val_batches and self.cfg.validate_every_steps
                    and self._step % self.cfg.validate_every_steps == 0):
                vb = val_batches
                if self.cfg.intra_val_max_batches > 0:
                    vb = list(val_batches)[: self.cfg.intra_val_max_batches]
                try:
                    metrics = self.validate(vb)  # appends to history.validations
                    if self.verbose:
                        print(f"  [step {self._step}] {metrics.one_line()}")
                finally:
                    # validate() leaves the model in eval mode; training must
                    # continue in train mode regardless of what validate did.
                    self.model.train()
                if self.cfg.checkpoint_history and self.dist.is_main:
                    self.history.to_json(
                        os.path.join(self.cfg.out_dir, "history.json")
                    )
                # Refresh figures so the fresh validation point shows up even
                # when step-plotting is not separately enabled (rank 0 only).
                self._write_plots()
            if (self.cfg.plot_every_steps
                    and self._step % self.cfg.plot_every_steps == 0):
                self._write_plots()

        summary = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "seconds": time.time() - started,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        self.history.epochs.append(summary)
        return summary

    @torch.no_grad()
    def validate(self, batches: Sequence[tuple]) -> ValidationMetrics:
        """Evaluate on held-out reads, in aligner terms rather than loss alone."""
        self.model.eval()
        losses, terms = [], {}
        a_auc, a_prec, a_rec, c_acc, q_mae = [], [], [], [], []
        all_mapq, all_correct = [], []
        n_chain_scored = 0

        for sup, reference in progress(
            self._feed(batches),
            total=len(batches),
            desc="validate",
            unit="batch",
            disable=not self.verbose,
            leave=False,
        ):
            with self.accel.precision():
                outputs, seed_scores, chain_scores = self._forward(sup, reference)
                loss = self.criterion(
                    outputs, sup.targets,
                    seed_scores=seed_scores, chain_scores=chain_scores,
                )
            losses.append(float(loss))
            for key, value in loss.terms.items():
                terms.setdefault(key, []).append(value)

            am = anchor_metrics(
                seed_scores["logits"], sup.targets["seed_labels"],
                sup.targets.get("anchor_mask"),
            )
            a_auc.append(am["auc"])
            a_prec.append(am["precision"])
            a_rec.append(am["recall"])
            acc, n_scored = chain_accuracy(
                chain_scores["logits"], sup.targets["chain_target"],
                sup.targets.get("chain_mask"),
            )
            if n_scored:
                c_acc.append(acc)
                n_chain_scored += n_scored

            mapping = getattr(outputs, "mapping", None)
            if isinstance(mapping, dict) and "mapq" in mapping:
                predicted = mapping["mapq"].flatten()
                q_mae.append(mapq_mae(predicted, sup.targets["mapq_target"]))
                all_mapq.extend(predicted.detach().float().cpu().tolist())
                # "Correct" for calibration = the re-ranker picked the right chain.
                scores = chain_scores["logits"]
                if scores.dim() == 3:
                    scores = scores.squeeze(-1)
                picked = scores.argmax(-1)
                all_correct.extend(
                    (picked == sup.targets["chain_target"]).detach().cpu().tolist()
                )

        calib = mapq_calibration(all_mapq, all_correct)
        placement = self._locus_report(batches)
        metrics = ValidationMetrics(
            loss=float(np.mean(losses)) if losses else 0.0,
            terms={k: float(np.mean(v)) for k, v in terms.items()},
            locus_accuracy=placement["locus_accuracy"],
            chain_accuracy=float(np.mean(c_acc)) if c_acc else 0.0,
            anchor_auc=float(np.mean(a_auc)) if a_auc else 0.0,
            anchor_precision=float(np.mean(a_prec)) if a_prec else 0.0,
            anchor_recall=float(np.mean(a_rec)) if a_rec else 0.0,
            mapq_mae=float(np.mean(q_mae)) if q_mae else 0.0,
            mapq_expected_error=calib["expected_error"],
            mapq_observed_error=calib["observed_error"],
            mapped_fraction=placement["mapped_fraction"],
            n_reads=sum(len(reads) for reads, _ in batches),
            n_chain_scored=n_chain_scored,
        )
        # Under DDP each rank validated its own shard; reduce to one global set
        # of numbers so the logged metrics *and* the early-stopping decision are
        # identical on every rank (ranks must agree or they dead-lock on the
        # next collective). A no-op in a single-process run.
        metrics = self._reduce_metrics(metrics)
        self.history.validations.append(metrics)
        return metrics

    def _reduce_metrics(self, m: ValidationMetrics) -> ValidationMetrics:
        """All-reduce a :class:`ValidationMetrics` across ranks (mean/sum).

        Rate-style fields (loss, accuracies, calibration) are averaged over the
        ranks; count-style fields (``n_reads``, ``n_chain_scored``) are summed.
        Returns ``m`` unchanged when not distributed.
        """
        if not self.dist.enabled:
            return m
        mean = self.dist.reduce_mean
        m.loss = mean(m.loss)
        m.terms = {k: mean(v) for k, v in m.terms.items()}
        m.locus_accuracy = mean(m.locus_accuracy)
        m.chain_accuracy = mean(m.chain_accuracy)
        m.anchor_auc = mean(m.anchor_auc)
        m.anchor_precision = mean(m.anchor_precision)
        m.anchor_recall = mean(m.anchor_recall)
        m.mapq_mae = mean(m.mapq_mae)
        m.mapq_expected_error = mean(m.mapq_expected_error)
        m.mapq_observed_error = mean(m.mapq_observed_error)
        m.mapped_fraction = mean(m.mapped_fraction)
        m.n_reads = int(round(self.dist.reduce_sum(m.n_reads)))
        m.n_chain_scored = int(round(self.dist.reduce_sum(m.n_chain_scored)))
        return m

    @torch.no_grad()
    def _locus_report(self, batches: Sequence[tuple]) -> dict[str, float]:
        """Run the real aligner and check where it actually placed each read.

        This is the end-to-end number: it exercises seeding, chaining, the neural
        scoring and extension together, so it moves only when the model makes the
        whole pipeline place reads better -- unlike the per-head metrics above.

        Eval mode is forced here rather than assumed. Left in train mode, dropout
        perturbs the seed scores enough to shift placement by thousands of bases,
        which would show up as a mysteriously bad metric rather than as a bug.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            predicted, truth, mapped = [], [], 0
            for reads, reference in progress(
                batches, desc="locus report", unit="batch", leave=False
            ):
                results, _ = self.pipeline.align(reads, reference)
                for read, alignments in zip(reads, results):
                    record = alignments.records[0] if alignments.records else None
                    if record is None or not record.is_mapped:
                        continue
                    mapped += 1
                    predicted.append(record.ref_start)
                    truth.append(read.ref_start)
        finally:
            self.model.train(was_training)

        n_total = sum(len(reads) for reads, _ in batches)
        return {
            "locus_accuracy": locus_accuracy(predicted, truth),
            "mapped_fraction": mapped / max(n_total, 1),
        }

    # ---- driver ------------------------------------------------------------ #
    def fit(self, train_batches: Sequence[tuple],
            val_batches: Sequence[tuple],
            resume: Optional[str] = None) -> TrainHistory:
        """Train with early stopping on the monitored validation metric.

        Both batch arguments are sequences of ``(reads, reference)`` pairs.

        ``resume`` (a checkpoint path, or ``"auto"`` to pick the latest one under
        ``out_dir``) restores model/optimizer/step and continues toward the same
        ``--epochs`` *total* target, so the remaining epochs are trained. Left at
        ``None`` (the default) the run starts from scratch, byte-for-byte as
        before.
        """
        torch.manual_seed(self.cfg.seed)
        # Data-parallel sharding: each rank trains on its own strided, equal-size
        # slice of the pre-built batch list (rank i -> batches[i::world_size]).
        # A no-op single process, so single-GPU sees the full list as before.
        train_batches = self.dist.shard(train_batches)
        val_batches = self.dist.shard(val_batches)
        accum = max(1, int(getattr(self.cfg, "grad_accum", 1)))
        steps_per_epoch = max(1, math.ceil(len(train_batches) / accum))
        total_steps = max(1, self.cfg.epochs * steps_per_epoch)
        best, best_epoch, stale = -math.inf, -1, 0
        best_state: Optional[dict] = None
        warned_monitor = False

        # Resume restores weights/optimizer/step and returns where to continue.
        # When not resuming this is skipped entirely (start_epoch stays 0).
        start_epoch = 0
        if resume:
            start_epoch, best, best_epoch = self._load_resume(resume, steps_per_epoch)

        if self.verbose:
            n_train = sum(len(reads) for reads, _ in train_batches)
            n_val = sum(len(reads) for reads, _ in val_batches)
            print(f"device: {self.accel.summary()}")
            print(f"training {sum(p.numel() for p in self.model.parameters()):,} "
                  f"parameters on {n_train} reads ({n_val} held out) "
                  f"for up to {self.cfg.epochs} epochs"
                  + (f" (resuming at epoch {start_epoch})" if start_epoch else "")
                  + "\n")

        epoch_bar = progress(
            range(start_epoch, self.cfg.epochs),
            desc="epochs",
            unit="epoch",
            disable=not self.verbose,
        )
        # Optionally cap the epoch-boundary validation for speed (0 = full pass).
        epoch_val = val_batches
        cap = int(getattr(self.cfg, "epoch_val_max_batches", 0) or 0)
        if cap > 0 and len(val_batches) > cap:
            epoch_val = list(val_batches)[:cap]
            if self.verbose:
                print(f"epoch-end validation capped to {cap} of {len(val_batches)} "
                      "batches (TrainConfig.epoch_val_max_batches; 0 = full pass)")

        for epoch in epoch_bar:
            summary = self.train_epoch(train_batches, epoch, total_steps,
                                       val_batches)
            metrics = self.validate(epoch_val)

            score = metrics.monitored(self.cfg.monitor)
            if score is None:
                if not warned_monitor:
                    warned_monitor = True
                    print(f"warning: {self.cfg.monitor!r} is not measurable on this "
                          f"data (needs >=2 candidate chains per read); early "
                          f"stopping will use -val_loss instead")
                score = -metrics.loss

            if self.verbose:
                print(f"epoch {epoch:02d}  train={summary['train_loss']:.4f}  "
                      f"{metrics.one_line()}  ({summary['seconds']:.1f}s)")
                if hasattr(epoch_bar, "set_postfix"):
                    epoch_bar.set_postfix(
                        train=f"{summary['train_loss']:.4f}",
                        monitor=f"{score:.4f}",
                        refresh=False,
                    )

            improved = score > best
            if improved:
                best, best_epoch, stale = score, epoch, 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.raw_model.state_dict().items()
                }

            # Persist *every* epoch so nothing is lost mid-run: a per-epoch
            # checkpoint, a rolling ``last.pt``, and a flushed history.json.
            self._persist_epoch(epoch=epoch, score=score, is_best=improved,
                                best_epoch=best_epoch, best_score=best)

            if not improved:
                stale += 1
                if stale >= self.cfg.patience:
                    if self.verbose:
                        print(f"early stop: {self.cfg.monitor} has not improved "
                              f"for {stale} epochs (best {best:.4f} @ epoch "
                              f"{best_epoch})")
                    break

        if best_state is not None:
            self.raw_model.load_state_dict(best_state)
            # Every rank holds identical best weights (same init seed + synced
            # gradients), so only rank 0 writes the checkpoint to disk.
            if self.cfg.save_checkpoint and self.dist.is_main:
                ckpt_path = self.save_checkpoint(
                    best_epoch=best_epoch, best_score=best
                )
                if self.verbose:
                    print(f"checkpoint -> {ckpt_path}")

        self.probe.close()
        # Hold the other ranks at the finish line until rank 0 has written the
        # final checkpoint/history/plots, so no rank tears down the process group
        # (and with it NCCL) while writes are still in flight.
        self.dist.barrier()
        if self.verbose:
            print(f"\nbest {self.cfg.monitor}={best:.4f} at epoch {best_epoch}")
        return self.history

    def _write_plots(self) -> None:
        """Redraw the run's figures into ``out_dir/plots`` (best-effort).

        Plotting is reporting: it must never abort training, so every failure is
        swallowed exactly like the per-epoch plotting in :meth:`_persist_epoch`.
        """
        if not self.dist.is_main:
            return
        try:
            from .plots import plot_all

            plot_all(self.history, os.path.join(self.cfg.out_dir, "plots"),
                     verbose=False)
        except Exception as exc:  # noqa: BLE001 - plotting is best-effort
            if self.verbose:
                print(f"  (intra-epoch plot skipped: {exc})")

    # ---- resume ------------------------------------------------------------ #
    def _latest_checkpoint(self) -> Optional[str]:
        """Newest resumable checkpoint under ``out_dir`` (for ``--resume auto``).

        Considers the rolling ``last.pt`` plus every ``checkpoints/epoch_*.pt``
        and ``checkpoints/step_*.pt``, and returns the most recently modified.
        ``None`` when the run directory holds no checkpoint yet.
        """
        import glob

        out = self.cfg.out_dir
        ckpt_dir = os.path.join(out, "checkpoints")
        cands = (
            glob.glob(os.path.join(ckpt_dir, "epoch_*.pt"))
            + glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
        )
        last = os.path.join(out, "last.pt")
        if os.path.exists(last):
            cands.append(last)
        cands = [c for c in cands if os.path.exists(c)]
        if not cands:
            return None
        return max(cands, key=os.path.getmtime)

    def _restore_history(self, resume_step: int, next_epoch: int) -> None:
        """Reload the existing ``history.json`` so a resumed run appends to it.

        Steps/epochs at or beyond the resume point are dropped so a re-done
        (interrupted) epoch does not duplicate entries; ``StepReport`` and
        ``ValidationMetrics`` are flat dataclasses, so ``**dict`` round-trips
        them exactly. On any parse failure the old log is preserved beside the
        run (``history.pre_resume.json``) and a fresh in-memory log is started,
        so a corrupt history can never abort a resume.
        """
        path = os.path.join(self.cfg.out_dir, "history.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            self.history.steps = [
                StepReport(**s) for s in data.get("steps", [])
                if int(s.get("step", 0)) < resume_step
            ]
            self.history.epochs = [
                e for e in data.get("epochs", [])
                if int(e.get("epoch", 0)) < next_epoch
            ]
            self.history.validations = [
                ValidationMetrics(**v) for v in data.get("validations", [])
            ]
        except Exception as exc:  # noqa: BLE001 - a bad log must never abort resume
            backup = os.path.join(self.cfg.out_dir, "history.pre_resume.json")
            try:
                os.replace(path, backup)
            except OSError:
                backup = path
            if self.verbose:
                print(f"resume: WARNING could not parse {path} ({exc}); kept it at "
                      f"{backup} and started a fresh history log")

    def _load_resume(self, resume: str,
                     steps_per_epoch: int) -> tuple[int, float, int]:
        """Restore state from a checkpoint; return ``(start_epoch, best, best_epoch)``.

        Robust to checkpoints that predate the resume fields: a missing optimizer
        state starts the optimizer fresh (with a warning), and a missing step
        counter is derived from the epoch boundary so the LR schedule still lines
        up. A per-epoch checkpoint resumes at the next epoch; a mid-epoch (step)
        checkpoint re-does its in-progress epoch from the start.
        """
        if resume == "auto":
            path = self._latest_checkpoint()
            if path is None:
                if self.verbose:
                    print(f"resume: no checkpoint under {self.cfg.out_dir}; "
                          "starting a fresh run")
                return 0, -math.inf, -1
        else:
            path = resume
            if not os.path.exists(path):
                raise FileNotFoundError(f"--resume checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Model weights (strict=False so a checkpoint from a slightly different
        # build still loads everything it can).
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = self.raw_model.load_state_dict(state, strict=False)
        if self.verbose and (missing or unexpected):
            print(f"resume: model loaded with missing={len(missing)} "
                  f"unexpected={len(unexpected)} keys")

        # Criterion (learnable Kendall weights) — best-effort.
        if isinstance(ckpt, dict) and ckpt.get("loss") is not None:
            try:
                self.criterion.load_state_dict(ckpt["loss"])
            except Exception as exc:  # noqa: BLE001 - keep resuming on mismatch
                if self.verbose:
                    print(f"resume: WARNING could not load loss weights ({exc})")

        # Optimizer — fall back to a fresh optimizer if absent/incompatible.
        opt_loaded = False
        if isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                opt_loaded = True
            except Exception as exc:  # noqa: BLE001 - keep resuming on mismatch
                if self.verbose:
                    print(f"resume: WARNING optimizer state incompatible ({exc}); "
                          "starting the optimizer fresh")
        elif self.verbose:
            print("resume: checkpoint has no optimizer state; starting it fresh")

        # Where to continue. Old checkpoints predate ``epoch_completed``: a
        # per-epoch file has no ``step`` key, a step file does — infer from that.
        epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
        if isinstance(ckpt, dict) and "epoch_completed" in ckpt:
            completed = bool(ckpt["epoch_completed"])
        else:
            completed = not (isinstance(ckpt, dict) and "step" in ckpt)
        next_epoch = max(0, epoch + 1 if completed else epoch)
        # Align the global step to the epoch boundary so the LR schedule is
        # exactly what an uninterrupted run would have produced at this point.
        self._step = next_epoch * steps_per_epoch

        best, best_epoch = -math.inf, -1
        if isinstance(ckpt, dict):
            bs = ckpt.get("best_score")
            be = int(ckpt.get("best_epoch", -1))
            if be >= 0 and isinstance(bs, (int, float)) and math.isfinite(bs):
                best, best_epoch = float(bs), be

        # Append to (never clobber) the existing history — only rank 0 owns it.
        if self.dist.is_main:
            self._restore_history(self._step, next_epoch)

        if self.verbose:
            extra = ", optimizer restored" if opt_loaded else ", optimizer fresh"
            if best_epoch >= 0:
                extra += f", best {self.cfg.monitor}={best:.4f}@epoch{best_epoch}"
            print(f"resume: {path}")
            print(f"resume: continuing at epoch {next_epoch} "
                  f"(global step {self._step}{extra})")
        return next_epoch, best, best_epoch

    def _checkpoint_payload(self, *, epoch: int, best_epoch: int,
                            best_score: float, epoch_completed: bool = True) -> dict:
        payload = {
            "model": self.raw_model.state_dict(),
            "loss": self.criterion.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
            "epoch": epoch,
            # Global optimizer-step counter, saved in *every* checkpoint so a
            # resumed run restores the LR schedule position (the LR is derived
            # purely from this step; there is no separate scheduler object).
            "step": self._step,
            # True when this checkpoint sits on an epoch boundary (a per-epoch or
            # final-best save); False for a mid-epoch step checkpoint. Resume uses
            # it to decide whether to advance to the next epoch or re-do the
            # in-progress one.
            "epoch_completed": epoch_completed,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "monitor": self.cfg.monitor,
            "device_summary": self.history.device_summary or self.accel.summary(),
            "device_ids": list(self.device_ids),
        }
        # Preserve model architecture knobs when present (eval needs d_model).
        if hasattr(self.raw_model, "cfg"):
            try:
                payload["model_cfg"] = asdict(self.raw_model.cfg)
            except TypeError:
                payload["model_cfg"] = None
        return payload

    def _persist_step(self, *, epoch: int) -> None:
        """Save an intra-epoch checkpoint at the current global step.

        Written to ``checkpoints/step_XXXXXX.pt`` alongside the per-epoch files.
        Carries the current step so a resumed run knows how far it got; best-so-
        far bookkeeping stays with the epoch-level ``checkpoint.pt``.
        """
        if not self.dist.is_main:
            return
        ckpt_dir = os.path.join(self.cfg.out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        payload = self._checkpoint_payload(
            epoch=epoch, best_epoch=-1, best_score=float("nan"),
            epoch_completed=False,
        )
        step_path = os.path.join(ckpt_dir, f"step_{self._step:06d}.pt")
        torch.save(payload, step_path)
        # Also refresh last.pt so the newest state is always one file away.
        if self.cfg.save_last:
            torch.save(payload, os.path.join(self.cfg.out_dir, "last.pt"))
        # Rotate step checkpoints so frequent saving cannot fill the disk: keep
        # only the most recent ``keep_step_checkpoints`` (default 5). last.pt and
        # the per-epoch checkpoints are untouched, so nothing durable is lost.
        import contextlib
        import glob
        keep = int(getattr(self.cfg, "keep_step_checkpoints", 5) or 0)
        if keep > 0:
            existing = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
            for stale in existing[:-keep]:
                with contextlib.suppress(OSError):
                    os.remove(stale)
        if self.verbose:
            print(f"  saved {step_path}")

    def _persist_epoch(self, *, epoch: int, score: float, is_best: bool,
                       best_epoch: int, best_score: float) -> None:
        """Save per-epoch checkpoint, rolling ``last.pt`` and the history log.

        Nothing here gates on ``save_checkpoint`` (that flag is specifically the
        final *best* weights): this is the "never lose a step" bookkeeping the
        user asked for, so every epoch is recoverable and inspectable.
        """
        # Only rank 0 owns the run directory; other ranks hold identical state.
        if not self.dist.is_main:
            return
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        payload = self._checkpoint_payload(
            epoch=epoch, best_epoch=best_epoch, best_score=best_score
        )

        if self.cfg.save_every_epoch:
            ckpt_dir = os.path.join(self.cfg.out_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ep_path = os.path.join(ckpt_dir, f"epoch_{epoch:02d}.pt")
            torch.save(payload, ep_path)
            if self.verbose:
                tag = "  (best)" if is_best else ""
                print(f"  saved {ep_path}{tag}")

        if self.cfg.save_last:
            torch.save(payload, os.path.join(self.cfg.out_dir, "last.pt"))

        if self.cfg.checkpoint_history:
            self.history.to_json(os.path.join(self.cfg.out_dir, "history.json"))

        # Refresh the figures each epoch so the loss/validation curves exist
        # mid-run (and after a crash), not only once fit() returns. Plotting is
        # reporting: never let it break the training loop.
        if self.cfg.plot_every_epoch:
            try:
                from .plots import plot_all

                plot_all(self.history, os.path.join(self.cfg.out_dir, "plots"),
                         verbose=False)
            except Exception as exc:  # noqa: BLE001 - plotting is best-effort
                if self.verbose:
                    print(f"  (per-epoch plot skipped: {exc})")

    def save_checkpoint(self, *, best_epoch: int = -1,
                        best_score: float = float("nan"),
                        filename: str = "checkpoint.pt") -> str:
        """Write best weights + training metadata under ``out_dir``."""
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        path = os.path.join(self.cfg.out_dir, filename)
        payload = self._checkpoint_payload(
            epoch=best_epoch, best_epoch=best_epoch, best_score=best_score
        )
        torch.save(payload, path)
        return path
