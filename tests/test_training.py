"""Tests for the training stack: targets, metrics, probes, trainer, plots.

The failures worth guarding against here are the quiet ones -- a run that looks
healthy but has learned nothing:

  * a NaN feature silently poisons every loss from step 0 (``AnchorSet.score`` is
    NaN until the scorer fills it, and feeding it to the head did exactly this);
  * labels drawn at random still produce a falling loss curve, so the anchor
    labels are checked against the true diagonal;
  * chain accuracy reads 100% when each read has one candidate, which then makes
    early stopping fire on a metric that can never move;
  * a validation metric measured in train mode is degraded by dropout.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import torch

from graphmambaformer.alignment.pipeline import build_pipeline
from graphmambaformer.config import (
    CoreModelConfig,
    GraphMambaConfig,
    LossConfig,
    PipelineConfig,
)
from graphmambaformer.data.synthetic import generate_dataset, preset
from graphmambaformer.models import build_core_model
from graphmambaformer.training import (
    BehaviorProbe,
    TargetBuilder,
    TrainConfig,
    Trainer,
    ValidationMetrics,
    anchor_metrics,
    chain_accuracy,
    locus_accuracy,
    mapq_calibration,
    plot_all,
    plotting_available,
)


def _fixture(d_model: int = 64):
    torch.manual_seed(0)
    ds = generate_dataset(preset("tiny"))
    ref = ds.references[0]
    reads = [r for r in ds.splits["train"] if r.ref_id == 0]
    cfg = GraphMambaConfig(d_model=d_model)
    model = build_core_model(CoreModelConfig(arch="graphmamba", graphmamba=cfg)).model
    pipe = build_pipeline(PipelineConfig(mode="hybrid", batch_size=4), model=model)
    reference = pipe.build_reference(ref.seq, ref_id=0)
    return ds, ref, reads, model, pipe, reference


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def test_supervision_tensors_are_finite():
    """Regression: AnchorSet.score is NaN, and it used to reach the loss."""
    _, _, reads, model, pipe, reference = _fixture()
    sup = TargetBuilder(pipe, model=model).build(reads, reference)

    for name in ("seed_features", "chain_feats"):
        t = getattr(sup, name)
        assert torch.isfinite(t).all(), f"{name} contains NaN/inf"
    for key, value in sup.targets.items():
        if value.is_floating_point():
            assert torch.isfinite(value).all(), f"target {key} contains NaN/inf"
    print(f"all supervision tensors finite "
          f"(seed_features {tuple(sup.seed_features.shape)}, "
          f"{len(sup.targets)} target keys)")


def test_anchor_labels_track_the_true_diagonal():
    """Labels must come from the truth, not be arbitrary -- else nothing is learned."""
    _, _, reads, model, pipe, reference = _fixture()
    builder = TargetBuilder(pipe, model=model)
    sup = builder.build(reads, reference)

    balance = TargetBuilder.label_balance(sup)
    assert 0.0 < balance["anchor_positive_rate"] < 1.0, balance
    assert balance["anchors_per_read"] > 0

    # An anchor labelled positive must really imply the read's true locus.
    anchors = pipe.seed([r.seq for r in reads], reference)
    for row, read in enumerate(reads):
        a = anchors[row]
        if len(a) == 0:
            continue
        labels = builder._anchor_labels(a, read.ref_start, len(read.seq))
        implied = a.ref_pos.astype(np.int64) - a.read_pos.astype(np.int64)
        for i, lab in enumerate(labels):
            if lab > 0.5:
                assert abs(int(implied[i]) - read.ref_start) <= 20
    print(f"anchor labels agree with the true diagonal; "
          f"positive rate {balance['anchor_positive_rate']:.1%}")


def test_shuffled_labels_are_detectably_different():
    """Guards the test above: random labels give ~50% positives and no structure."""
    _, _, reads, model, pipe, reference = _fixture()
    sup = TargetBuilder(pipe, model=model).build(reads, reference)
    real = sup.targets["seed_labels"]
    fake = torch.randint(0, 2, real.shape).float()
    assert not torch.allclose(real, fake), "real labels should not match random"
    print(f"real positive rate {float(real.mean()):.2f} vs random "
          f"{float(fake.mean()):.2f} - labels carry signal")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_chain_accuracy_refuses_single_candidate_reads():
    """Picking 1 of 1 is not a measurement; it must report n=0, not 100%."""
    logits = torch.tensor([[2.0], [3.0]])
    mask = torch.ones(2, 1, dtype=torch.bool)
    target = torch.tensor([0, 0])
    acc, n = chain_accuracy(logits, target, mask)
    assert n == 0, f"single-candidate reads must not be scored, got n={n}"

    logits2 = torch.tensor([[0.1, 5.0], [5.0, 0.1]])
    mask2 = torch.ones(2, 2, dtype=torch.bool)
    acc2, n2 = chain_accuracy(logits2, torch.tensor([1, 0]), mask2)
    assert (acc2, n2) == (1.0, 2), (acc2, n2)
    print("chain accuracy scores only reads with >=2 candidates (n=0 vs n=2)")


def test_unmeasurable_monitor_falls_back_instead_of_early_stopping():
    """Regression: monitoring chain_accuracy stopped every run at epoch 0."""
    m = ValidationMetrics(loss=1.5, chain_accuracy=0.0, n_chain_scored=0,
                          locus_accuracy=0.75)
    assert m.monitored("chain_accuracy") is None, "must report unmeasurable"
    assert m.monitored("locus_accuracy") == 0.75

    measurable = ValidationMetrics(chain_accuracy=0.5, n_chain_scored=10)
    assert measurable.monitored("chain_accuracy") == 0.5
    print("unmeasurable monitor returns None (so the trainer can fall back); "
          "measurable one returns its value")


def test_anchor_auc_is_chance_on_random_scores():
    labels = torch.cat([torch.ones(64), torch.zeros(64)])
    perfect = anchor_metrics(torch.cat([torch.ones(64) * 9, -torch.ones(64) * 9]), labels)
    assert perfect["auc"] > 0.99, perfect
    torch.manual_seed(0)
    chance = anchor_metrics(torch.randn(128), labels)
    assert 0.35 < chance["auc"] < 0.65, chance
    print(f"anchor AUC: perfect={perfect['auc']:.2f} random={chance['auc']:.2f}")


def test_mapq_calibration_detects_overconfidence():
    """An over-confident mapper must show observed error above what it promises."""
    honest = mapq_calibration([3] * 100, [True] * 50 + [False] * 50)
    over = mapq_calibration([60] * 100, [True] * 50 + [False] * 50)
    assert over["observed_error"] > over["expected_error"], over
    assert over["expected_error"] < 1e-3, over
    print(f"MAPQ 60 with 50% errors -> promises {over['expected_error']:.1e}, "
          f"observes {over['observed_error']:.2f} (over-confident); "
          f"MAPQ 3 promises {honest['expected_error']:.2f}")


def test_locus_accuracy_uses_tolerance():
    assert locus_accuracy([100, 200], [100, 200]) == 1.0
    assert locus_accuracy([100], [130]) == 1.0     # within 50
    assert locus_accuracy([100], [400]) == 0.0
    print("locus accuracy honours the 50 bp tolerance")


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def test_probe_sees_activations_gradients_and_routing():
    _, _, reads, model, pipe, reference = _fixture()
    sup = TargetBuilder(pipe, model=model).build(reads, reference)
    sup = sup.to(next(model.parameters()).device)
    probe = BehaviorProbe(model)
    try:
        outputs = model(sup.base_codes, mask=sup.mask, graph=reference.graph,
                        qualities=sup.qualities, modality=sup.modality)
        from graphmambaformer.losses import GraphMambaLoss
        loss = GraphMambaLoss(LossConfig())(outputs, sup.targets)
        loss.total.backward()

        report = probe.report(0, 0, loss, outputs)
        assert report.activations, "no activations captured"
        assert report.grad_norm_total > 0, "gradients did not flow"
        assert report.router_distribution, "router distribution missing"
        assert abs(sum(report.router_distribution.values()) - 1.0) < 1e-4
        # A live tower has spread; all-zero std would mean a dead tower.
        alive = [k for k, v in report.activations.items() if v["std"] > 0]
        assert alive, report.activations
        # The sequence path must always be observed. A typo in a tower name
        # previously left towers hooked but silently never reporting.
        for tower in ("sequence_encoder", "mamba_tower"):
            assert tower in report.activations, (tower, list(report.activations))
        # This reference is linear, so the graph towers correctly stay idle --
        # see the next test for the graph case.
        assert "gat_tower" not in report.activations
        print(f"probe: {len(report.activations)} towers ({len(alive)} alive), "
              f"|g|={report.grad_norm_total:.3f}, "
              f"routes={report.router_distribution}")
        print("   " + report.one_line())
    finally:
        probe.close()


def test_probe_reports_graph_towers_when_a_graph_is_given():
    """With a graph present, the GAT and graph-encoder towers must report too."""
    from graphmambaformer.models.graph_mamba import GraphBatch

    _, _, reads, model, pipe, reference = _fixture()
    device = next(model.parameters()).device
    sup = TargetBuilder(pipe, model=model).build(reads, reference).to(device)

    cfg = model.cfg
    torch.manual_seed(0)
    edges = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    graph = GraphBatch(
        node_kmer_ids=torch.randint(0, 4 ** cfg.graph_encoder.kmer_size, (6, 5)),
        edge_index=edges,
        edge_type=torch.zeros(edges.shape[1], dtype=torch.long),
    ).to(device)

    probe = BehaviorProbe(model)
    try:
        outputs = model(sup.base_codes, mask=sup.mask, graph=graph)
        report = probe.report(0, 0, type("L", (), {"terms": {}, "weights": {},
                                                   "__float__": lambda s: 0.0})(),
                              outputs, with_gradients=False)
        for tower in ("graph_encoder", "gat_tower", "fusion"):
            assert tower in report.activations, (tower, list(report.activations))
        print(f"with a graph, all {len(report.activations)} towers report: "
              f"{sorted(report.activations)}")
    finally:
        probe.close()


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
def test_training_step_reduces_loss_and_keeps_it_finite():
    _, _, reads, model, pipe, reference = _fixture()
    trainer = Trainer(model, pipe, cfg=TrainConfig(epochs=1, lr=1e-3), verbose=False)
    batches = [(reads, reference)]

    first = trainer.train_epoch(batches, epoch=0, total_steps=6)
    assert np.isfinite(first["train_loss"]), first
    losses = [first["train_loss"]]
    for epoch in range(1, 4):
        losses.append(trainer.train_epoch(batches, epoch, 6)["train_loss"])
    assert all(np.isfinite(x) for x in losses), losses
    assert losses[-1] < losses[0], f"loss did not fall: {losses}"
    print(f"loss fell over 4 epochs: {losses[0]:.4f} -> {losses[-1]:.4f}")


def test_validation_forces_eval_mode():
    """Regression: measured in train mode, dropout moved placement by ~4.6 kb."""
    _, _, reads, model, pipe, reference = _fixture()
    trainer = Trainer(model, pipe, cfg=TrainConfig(epochs=1), verbose=False)
    model.train()
    report = trainer._locus_report([(reads, reference)])
    assert model.training, "the probe must restore the mode it found"
    assert 0.0 <= report["locus_accuracy"] <= 1.0
    assert report["mapped_fraction"] > 0
    print(f"locus report ran in eval mode and restored train mode; "
          f"locus={report['locus_accuracy']:.0%} "
          f"mapped={report['mapped_fraction']:.0%}")


def test_fit_records_history_and_plots():
    _, _, reads, model, pipe, reference = _fixture()
    trainer = Trainer(model, pipe,
                      cfg=TrainConfig(epochs=2, patience=5, lr=1e-3),
                      verbose=False)
    batches = [(reads, reference)]
    history = trainer.fit(batches, batches)

    assert len(history.validations) == 2, len(history.validations)
    assert history.steps and history.device_summary
    assert all(np.isfinite(s.total) for s in history.steps)

    d = tempfile.mkdtemp(prefix="gmf_train_")
    try:
        path = history.to_json(os.path.join(d, "history.json"))
        assert os.path.getsize(path) > 0
        if plotting_available():
            written = plot_all(history, os.path.join(d, "plots"), verbose=False)
            assert len(written) >= 4, written
            assert all(os.path.getsize(p) > 0 for p in written)
            print(f"history JSON + {len(written)} plots written")
        else:
            assert plot_all(history, os.path.join(d, "plots"), verbose=False) == []
            print("history JSON written; matplotlib absent so plots skipped cleanly")
    finally:
        shutil.rmtree(d, ignore_errors=True)
