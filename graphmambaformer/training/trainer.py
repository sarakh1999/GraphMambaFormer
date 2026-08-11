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

from ..accel import AccelContext
from ..config import AccelConfig, LossConfig
from ..device import resolve_device_ids, unwrap_model, wrap_data_parallel
from ..losses import GraphMambaLoss
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
    warmup_frac: float = 0.1
    #: Stop when the monitored metric has not improved for this many epochs.
    patience: int = 4
    #: Early-stopping metric. Falls back to -val_loss if it is unmeasurable on
    #: the data at hand (see ValidationMetrics.monitored).
    monitor: str = "locus_accuracy"
    log_every: int = 1
    seed: int = 0
    out_dir: str = "data/training_runs/latest"
    #: If set, write ``checkpoint.pt`` (best weights) under ``out_dir``.
    save_checkpoint: bool = True
    #: If set, also write ``checkpoints/epoch_XX.pt`` after every epoch, so no
    #: intermediate state is ever lost (the user can resume/inspect any epoch).
    save_every_epoch: bool = True
    #: If set, keep ``last.pt`` pointing at the most recent epoch's weights.
    save_last: bool = True
    #: If set, flush ``history.json`` to disk after every epoch (crash-safe log).
    checkpoint_history: bool = True
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
                 verbose: bool = True):
        self.cfg = cfg or TrainConfig()
        self.pipeline = pipeline
        self.verbose = verbose

        self.accel = accel or AccelContext(AccelConfig())
        self.device = self.accel.caps.device
        self.device_ids = resolve_device_ids(self.cfg.devices, primary=self.device)
        if self.device_ids and self.device.type in ("cuda", "xpu"):
            # Primary compute device is the first id in the multi-GPU list.
            self.device = torch.device(self.device.type, int(self.device_ids[0]))
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)

        model = model.to(self.device)
        self.model = wrap_data_parallel(model, self.device_ids)
        self.raw_model = unwrap_model(self.model)

        self.criterion = GraphMambaLoss(loss_cfg or LossConfig()).to(self.device)
        self.builder = TargetBuilder(pipeline, model=self.raw_model)
        self.probe = BehaviorProbe(self.raw_model)

        # Kendall log-variances are parameters too, so they must be optimized.
        params = list(self.raw_model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.scaler = self.accel.grad_scaler()
        summary = self.accel.summary()
        if len(self.device_ids) > 1:
            summary = f"{summary} | data_parallel=[{','.join(map(str, self.device_ids))}]"
        self.history = TrainHistory(
            device_summary=summary, config=asdict(self.cfg)
        )
        self._step = 0
        if self.verbose and len(self.device_ids) > 1:
            print(f"multi-GPU DataParallel on devices {self.device_ids} "
                  f"(primary {self.device})")

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
        )
        b, n_chain, n_members = sup.member_states_shape
        chain_scores = self.raw_model.score_chains(
            chain_features=sup.chain_feats,
            member_states=torch.zeros(
                b, n_chain, n_members, self.raw_model.cfg.d_model, device=self.device
            ),
            member_mask=torch.ones(
                b, n_chain, n_members, dtype=torch.bool, device=self.device
            ),
            chain_mask=sup.chain_mask,
        )
        return outputs, seed_scores, chain_scores

    def _lr_at(self, step: int, total: int) -> float:
        """Linear warmup then cosine decay."""
        warmup = max(1, int(total * self.cfg.warmup_frac))
        if step < warmup:
            return self.cfg.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return self.cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    # ---- epochs ------------------------------------------------------------ #
    def train_epoch(self, batches: Sequence[tuple], epoch: int,
                    total_steps: int) -> dict:
        """``batches`` is a sequence of ``(reads, reference)`` pairs.

        Carrying the reference per batch rather than once for the whole run is
        what lets a dataset with several references contribute all of its reads
        instead of only those on the first one.
        """
        self.model.train()
        losses: list[float] = []
        started = time.time()

        for reads, reference in batches:
            sup = self.builder.build(reads, reference).to(self.device)
            lr = self._lr_at(self._step, total_steps)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            with self.accel.autocast():
                outputs, seed_scores, chain_scores = self._forward(sup, reference)
                loss = self.criterion(
                    outputs, sup.targets,
                    seed_scores=seed_scores, chain_scores=chain_scores,
                )

            if self.scaler is not None:
                self.scaler.scale(loss.total).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.total.backward()

            # Report gradients after unscaling but before clipping, so the norms
            # describe what the model produced rather than what was allowed.
            report = self.probe.report(
                self._step, epoch, loss, outputs, split="train",
                seed_scores=seed_scores, chain_scores=chain_scores, lr=lr,
                label_balance=TargetBuilder.label_balance(sup),
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

            self.history.steps.append(report)
            losses.append(report.total)
            if self.verbose and self._step % self.cfg.log_every == 0:
                print("  " + report.one_line())
            # Flush the step log frequently so a crash never loses recent work.
            if self.cfg.checkpoint_history and (
                self._step % max(1, self.cfg.log_every) == 0
            ):
                self.history.to_json(os.path.join(self.cfg.out_dir, "history.json"))
            self._step += 1

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

        for reads, reference in batches:
            sup = self.builder.build(reads, reference).to(self.device)
            with self.accel.autocast():
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
        self.history.validations.append(metrics)
        return metrics

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
            for reads, reference in batches:
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
            val_batches: Sequence[tuple]) -> TrainHistory:
        """Train with early stopping on the monitored validation metric.

        Both arguments are sequences of ``(reads, reference)`` pairs.
        """
        torch.manual_seed(self.cfg.seed)
        total_steps = max(1, self.cfg.epochs * len(train_batches))
        best, best_epoch, stale = -math.inf, -1, 0
        best_state: Optional[dict] = None
        warned_monitor = False

        if self.verbose:
            n_train = sum(len(reads) for reads, _ in train_batches)
            n_val = sum(len(reads) for reads, _ in val_batches)
            print(f"device: {self.accel.summary()}")
            print(f"training {sum(p.numel() for p in self.model.parameters()):,} "
                  f"parameters on {n_train} reads ({n_val} held out) "
                  f"for up to {self.cfg.epochs} epochs\n")

        for epoch in range(self.cfg.epochs):
            summary = self.train_epoch(train_batches, epoch, total_steps)
            metrics = self.validate(val_batches)

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
            if self.cfg.save_checkpoint:
                ckpt_path = self.save_checkpoint(
                    best_epoch=best_epoch, best_score=best
                )
                if self.verbose:
                    print(f"checkpoint -> {ckpt_path}")

        self.probe.close()
        if self.verbose:
            print(f"\nbest {self.cfg.monitor}={best:.4f} at epoch {best_epoch}")
        return self.history

    def _checkpoint_payload(self, *, epoch: int, best_epoch: int,
                            best_score: float) -> dict:
        payload = {
            "model": self.raw_model.state_dict(),
            "loss": self.criterion.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
            "epoch": epoch,
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

    def _persist_epoch(self, *, epoch: int, score: float, is_best: bool,
                       best_epoch: int, best_score: float) -> None:
        """Save per-epoch checkpoint, rolling ``last.pt`` and the history log.

        Nothing here gates on ``save_checkpoint`` (that flag is specifically the
        final *best* weights): this is the "never lose a step" bookkeeping the
        user asked for, so every epoch is recoverable and inspectable.
        """
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
