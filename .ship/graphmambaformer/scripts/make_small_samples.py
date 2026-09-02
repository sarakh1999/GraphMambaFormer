"""Create small, CPU-runnable samples of every data source for pipeline testing.

Sources handled
----------------
1. Downloaded pangenome graph  (``scripts/downloaded_data/pangenome ref graph/
   hprc-v1.1-mc-chm13.gfa.gz``) — a ~9.5 GB gzipped GFA. We stream only the head
   of the file (the ``S``/segment section) and carve out a small, contiguous
   *reference-backbone* window of real CHM13 chr sequence, wiring consecutive
   reference segments with ``ref_link`` edges. (The variant ``L``-line section
   lives at the very end of the ~47 GB decompressed file, so bubble edges are
   omitted from this CPU sample — see the note printed at the end.)

2. Downloaded single-chromosome graph (``.../one chromosome ref graph/chr21.vg``)
   — vg's native binary/protobuf format. Converting it to GFA needs the ``vg``
   binary, which is not installed here, so we only write a short instructions
   note rather than a sample.

3. Constructed synthetic dataset (``data/synthetic_tiny.pt`` + the companion
   ``data/synthetic_tiny/graph.gfa``) — already tiny. We downsample it to a few
   reads per split, save a ``synthetic_small.pt`` bundle, and copy the synthetic
   GFA (which *does* carry the full 8 edge types / bubble topology).

Outputs land in ``data/small_samples/``.

Run: PYTHONPATH=. .venv/bin/python scripts/make_small_samples.py
"""

from __future__ import annotations

import gzip
import os
import shutil

from graphmambaformer.data import load_dataset, save_dataset, write_fastq
from graphmambaformer.data.synthetic import SyntheticDataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANGENOME_GFA_GZ = os.path.join(
    REPO, "scripts", "downloaded_data", "pangenome ref graph",
    "hprc-v1.1-mc-chm13.gfa.gz",
)
CHR_VG = os.path.join(
    REPO, "scripts", "downloaded_data", "one chromosome ref graph", "chr21.vg"
)
SYNTH_PT = os.path.join(REPO, "data", "synthetic_tiny.pt")
SYNTH_GFA = os.path.join(REPO, "data", "synthetic_tiny", "graph.gfa")
OUT_DIR = os.path.join(REPO, "data", "small_samples")


# --------------------------------------------------------------------------- #
# 1. Real pangenome-graph window (reference backbone)
# --------------------------------------------------------------------------- #
def _parse_tags(fields: list[str]) -> dict[str, str]:
    tags = {}
    for f in fields:
        parts = f.split(":", 2)
        if len(parts) == 3:
            tags[parts[0]] = parts[2]
    return tags


def extract_pangenome_window(
    n_nodes: int = 300, max_scan_lines: int = 400_000
) -> str | None:
    """Carve a contiguous reference-backbone window out of the HPRC GFA head."""
    if not os.path.exists(PANGENOME_GFA_GZ):
        print(f"[pangenome] not found: {PANGENOME_GFA_GZ} (skipping)")
        return None

    # Collect reference segments (SR:i:0) for the first contig we encounter.
    ref_nodes: dict[str, list[tuple[int, str]]] = {}  # SN -> [(SO, seq)]
    scanned = 0
    with gzip.open(PANGENOME_GFA_GZ, "rt") as fh:
        for line in fh:
            scanned += 1
            if scanned > max_scan_lines:
                break
            if not line.startswith("S\t"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            seq = fields[2]
            tags = _parse_tags(fields[3:])
            if tags.get("SR") != "0" or "SN" not in tags or "SO" not in tags:
                continue  # keep only reference-rank segments with coordinates
            if not seq or any(c not in "ACGT" for c in seq):
                continue
            ref_nodes.setdefault(tags["SN"], []).append((int(tags["SO"]), seq))

    if not ref_nodes:
        print("[pangenome] no reference segments found in scanned head (skipping)")
        return None

    # Choose the contig with the most collected reference segments.
    sn = max(ref_nodes, key=lambda k: len(ref_nodes[k]))
    nodes = sorted(ref_nodes[sn], key=lambda t: t[0])  # sort by reference offset

    # Take the longest run of segments that are adjacent on the reference
    # (offset_{i+1} == offset_i + len(seq_i)) so the window is truly contiguous.
    best_start, best_len = 0, 1
    run_start = 0
    for i in range(1, len(nodes)):
        prev_off, prev_seq = nodes[i - 1]
        off = nodes[i][0]
        if off == prev_off + len(prev_seq):
            if i - run_start + 1 > best_len:
                best_start, best_len = run_start, i - run_start + 1
        else:
            run_start = i
    window = nodes[best_start : best_start + min(n_nodes, best_len)]
    if len(window) < 2:  # fall back to first n_nodes by offset
        window = nodes[:n_nodes]

    out_path = os.path.join(OUT_DIR, "hprc_chm13_chr1_window.gfa")
    total_bp = sum(len(seq) for _, seq in window)
    with open(out_path, "w") as fh:
        fh.write("H\tVN:Z:1.0\n")
        for idx, (off, seq) in enumerate(window):
            fh.write(f"S\t{sn}_n{idx}\t{seq}\tLN:i:{len(seq)}\tRS:i:{off}\n")
        for idx in range(len(window) - 1):
            a, b = f"{sn}_n{idx}", f"{sn}_n{idx + 1}"
            fh.write(f"L\t{a}\t+\t{b}\t+\t0M\tzt:Z:ref_link\n")
    print(f"[pangenome] contig={sn}  nodes={len(window)}  bp={total_bp:,}  "
          f"span={window[0][0]:,}..{window[-1][0] + len(window[-1][1]):,}")
    print(f"[pangenome] wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# 2. chr21.vg — needs the `vg` binary
# --------------------------------------------------------------------------- #
def note_chr_vg() -> None:
    note = os.path.join(OUT_DIR, "chr21_vg_README.txt")
    have_vg = shutil.which("vg") is not None
    exists = os.path.exists(CHR_VG)
    with open(note, "w") as fh:
        fh.write("chr21.vg — single-chromosome pangenome graph (vg native format)\n")
        fh.write("=" * 68 + "\n")
        fh.write(f"source present: {exists} ({CHR_VG})\n")
        fh.write(f"`vg` on PATH:   {have_vg}\n\n")
        fh.write("vg files are protobuf/binary and cannot be parsed as GFA directly.\n")
        fh.write("Convert (and optionally sub-chunk) to GFA once `vg` is installed:\n\n")
        fh.write("  vg convert -f chr21.vg > chr21.gfa            # whole chromosome\n")
        fh.write("  vg chunk -x chr21.vg -p CHM13#chr21:1-200000 \\\n")
        fh.write("      | vg convert -f - > chr21_window.gfa      # a 200 kb window\n\n")
        fh.write("Then feed chr21_window.gfa through scripts/test_pipeline_stages.py\n")
        fh.write("(the same GFA reader used for the synthetic + pangenome samples).\n")
    print(f"[chr21.vg] vg present={have_vg}; wrote instructions -> {note}")


# --------------------------------------------------------------------------- #
# 3. Constructed synthetic dataset -> small bundle + reads + GFA copy
# --------------------------------------------------------------------------- #
def downsample_synthetic(per_split: int = 4) -> None:
    if not os.path.exists(SYNTH_PT):
        print(f"[synthetic] not found: {SYNTH_PT} (run generate_synthetic_data.py first)")
        return
    ds: SyntheticDataset = load_dataset(SYNTH_PT)
    small_splits = {k: list(v)[:per_split] for k, v in ds.splits.items()}
    small = SyntheticDataset(config=ds.config, references=ds.references, splits=small_splits)

    small_pt = os.path.join(OUT_DIR, "synthetic_small.pt")
    save_dataset(small, small_pt)
    sizes = {k: len(v) for k, v in small_splits.items()}
    print(f"[synthetic] downsampled splits {sizes} -> {small_pt}")

    # human-readable reads + copy the (full-topology) synthetic GFA
    reads_fastq = os.path.join(OUT_DIR, "reads_small.fastq")
    write_fastq(small.all_records(), reads_fastq)
    print(f"[synthetic] wrote {reads_fastq}")
    if os.path.exists(SYNTH_GFA):
        dst = os.path.join(OUT_DIR, "synthetic_graph.gfa")
        shutil.copyfile(SYNTH_GFA, dst)
        print(f"[synthetic] copied synthetic graph -> {dst}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 68)
    print("Creating small samples in", OUT_DIR)
    print("=" * 68)
    extract_pangenome_window()
    note_chr_vg()
    downsample_synthetic()
    print("-" * 68)
    print("done.")


if __name__ == "__main__":
    main()
