#!/usr/bin/env python3
"""Quantify how longcallD --refine-aln changed an aligner's read alignments.

longcallD re-aligns phased reads against a haplotype-aware MSA consensus, which
moves and merges indels the mapper placed inconsistently — mostly homopolymers
and tandem repeats. Two views are reported:

per-read   how many alignments had their CIGAR / indel structure rewritten.
per-locus  whether reads now agree on indel breakpoints. This is the metric that
           reflects polish: one true indel scattered across many nearby
           coordinates by the mapper should collapse onto a single (pos, type,
           len) locus supported by more reads. Watch the >= --min-sv-len row,
           where mapper disagreement is worst.

Edit distance (NM) is reported neutrally, not as better/worse: refinement
optimises haplotype consistency, not per-read edit distance, so NM can rise
slightly while a large indel moves to its correct breakpoint.

Only reads carrying HP/PS tags are refined; everything else is passed through
unchanged, so per-read counts are split by phased vs unphased (unphased acts as
a control and should show zero indel-structure change).

usage:
  refine_report.py --original aln.bam --refined refined.bam [--json out.json]
                   [--examples 5] [--min-sv-len 30]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphmambaformer.progress import progress

# pysam CIGAR op codes for the ops we score.
OP_INS = 1
OP_DEL = 2


@dataclass
class CigarStats:
    n_ins: int = 0
    n_del: int = 0
    ins_bases: int = 0
    del_bases: int = 0

    @classmethod
    def from_cigartuples(cls, cigartuples) -> "CigarStats":
        s = cls()
        for op, length in cigartuples or ():
            if op == OP_INS:
                s.n_ins += 1
                s.ins_bases += length
            elif op == OP_DEL:
                s.n_del += 1
                s.del_bases += length
        return s


@dataclass
class Totals:
    reads: int = 0
    cigar_changed: int = 0
    indel_structure_changed: int = 0
    start_moved: int = 0
    nm_decreased: int = 0
    nm_increased: int = 0
    d_n_ins: int = 0
    d_n_del: int = 0
    d_ins_bases: int = 0
    d_del_bases: int = 0
    d_nm: int = 0
    examples: list = field(default_factory=list)


@dataclass
class LocusStats:
    """Indel breakpoint agreement across reads, keyed by (pos, type, length)."""

    distinct_loci: int = 0
    total_ops: int = 0
    singleton_loci: int = 0
    mean_reads_per_locus: float = 0.0
    sv_distinct_loci: int = 0
    sv_total_ops: int = 0


def read_key(rec: pysam.AlignedSegment) -> Tuple[str, bool, bool]:
    """Identity of an alignment record that survives re-alignment.

    Refinement can shift reference_start, so position cannot be part of the key.
    Supplementary/secondary flags keep the split alignments of one read apart.
    """
    return (rec.query_name, rec.is_supplementary, rec.is_secondary)


def index_bam(path: str) -> Dict[Tuple[str, bool, bool], pysam.AlignedSegment]:
    out: Dict[Tuple[str, bool, bool], pysam.AlignedSegment] = {}
    with pysam.AlignmentFile(path, "rb", check_sq=False) as fh:
        for rec in progress(fh.fetch(until_eof=True),
                            desc=f"index {os.path.basename(path)}", unit="aln", leave=False):
            if rec.is_unmapped:
                continue
            out[read_key(rec)] = rec
    return out


def nm_of(rec: pysam.AlignedSegment) -> Optional[int]:
    try:
        return int(rec.get_tag("NM"))
    except (KeyError, ValueError):
        return None


def indel_loci(path: str, names: set, min_sv_len: int) -> LocusStats:
    """Tally indel breakpoints over the given read names.

    ``names`` is the phased set taken from the refined BAM, so the same reads are
    measured on both sides — HP/PS only exist after refinement.
    """
    loci: Counter = Counter()
    with pysam.AlignmentFile(path, "rb", check_sq=False) as fh:
        for rec in progress(fh.fetch(until_eof=True),
                            desc=f"indel loci {os.path.basename(path)}", unit="aln", leave=False):
            if rec.is_unmapped or rec.is_secondary or rec.is_supplementary:
                continue
            if rec.query_name not in names:
                continue
            pos = rec.reference_start
            for op, length in rec.cigartuples or ():
                if op in (0, 7, 8):          # M / = / X consume reference
                    pos += length
                elif op == OP_DEL:
                    loci[(pos, "D", length)] += 1
                    pos += length
                elif op == OP_INS:
                    loci[(pos, "I", length)] += 1

    total = sum(loci.values())
    n = len(loci)
    sv = {k: v for k, v in loci.items() if k[2] >= min_sv_len}
    return LocusStats(
        distinct_loci=n,
        total_ops=total,
        singleton_loci=sum(1 for v in loci.values() if v == 1),
        mean_reads_per_locus=(total / n) if n else 0.0,
        sv_distinct_loci=len(sv),
        sv_total_ops=sum(sv.values()),
    )


def phased_read_names(path: str) -> set:
    names = set()
    with pysam.AlignmentFile(path, "rb", check_sq=False) as fh:
        for rec in progress(fh.fetch(until_eof=True),
                            desc=f"phased reads {os.path.basename(path)}", unit="aln", leave=False):
            if rec.is_secondary or rec.is_supplementary:
                continue
            if rec.has_tag("HP"):
                names.add(rec.query_name)
    return names


def pct_delta(before: float, after: float) -> str:
    if not before:
        return "n/a"
    return f"{100.0 * (after - before) / before:+.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True, help="aligner BAM fed to longcallD")
    ap.add_argument("--refined", required=True, help="longcallD --refine-aln output BAM")
    ap.add_argument("--json", help="write the report as JSON here")
    ap.add_argument("--examples", type=int, default=5,
                    help="per-group example CIGAR rewrites to show (default 5)")
    ap.add_argument("--min-sv-len", type=int, default=30,
                    help="indel length counted as SV-size, matching longcallD -l "
                         "(default 30)")
    args = ap.parse_args()

    original = index_bam(args.original)
    refined = index_bam(args.refined)

    groups = {"phased": Totals(), "unphased": Totals()}
    missing = Counter()

    for key, ref_rec in progress(refined.items(), total=len(refined),
                                 desc="compare alignments", unit="aln"):
        orig_rec = original.get(key)
        if orig_rec is None:
            missing["only_in_refined"] += 1
            continue

        group = "phased" if ref_rec.has_tag("HP") else "unphased"
        t = groups[group]
        t.reads += 1

        o_cig, r_cig = orig_rec.cigarstring, ref_rec.cigarstring
        o_stats = CigarStats.from_cigartuples(orig_rec.cigartuples)
        r_stats = CigarStats.from_cigartuples(ref_rec.cigartuples)

        if o_cig != r_cig:
            t.cigar_changed += 1
            if len(t.examples) < args.examples:
                t.examples.append({
                    "read": ref_rec.query_name,
                    "ref_start_original": orig_rec.reference_start,
                    "ref_start_refined": ref_rec.reference_start,
                    "cigar_original": o_cig,
                    "cigar_refined": r_cig,
                })

        if (o_stats.n_ins, o_stats.n_del, o_stats.ins_bases, o_stats.del_bases) != \
           (r_stats.n_ins, r_stats.n_del, r_stats.ins_bases, r_stats.del_bases):
            t.indel_structure_changed += 1

        if orig_rec.reference_start != ref_rec.reference_start:
            t.start_moved += 1

        t.d_n_ins += r_stats.n_ins - o_stats.n_ins
        t.d_n_del += r_stats.n_del - o_stats.n_del
        t.d_ins_bases += r_stats.ins_bases - o_stats.ins_bases
        t.d_del_bases += r_stats.del_bases - o_stats.del_bases

        o_nm, r_nm = nm_of(orig_rec), nm_of(ref_rec)
        if o_nm is not None and r_nm is not None:
            t.d_nm += r_nm - o_nm
            if r_nm < o_nm:
                t.nm_decreased += 1
            elif r_nm > o_nm:
                t.nm_increased += 1

    only_orig = len(set(original) - set(refined))

    phased_names = phased_read_names(args.refined)
    loci_before = indel_loci(args.original, phased_names, args.min_sv_len)
    loci_after = indel_loci(args.refined, phased_names, args.min_sv_len)

    report = {
        "original_bam": args.original,
        "refined_bam": args.refined,
        "min_sv_len": args.min_sv_len,
        "alignments_original": len(original),
        "alignments_refined": len(refined),
        "only_in_original": only_orig,
        "only_in_refined": missing["only_in_refined"],
        "phased": asdict(groups["phased"]),
        "unphased": asdict(groups["unphased"]),
        "indel_loci_before": asdict(loci_before),
        "indel_loci_after": asdict(loci_after),
    }

    print(f"original : {args.original}")
    print(f"refined  : {args.refined}")
    print(f"alignments: {len(original)} original / {len(refined)} refined "
          f"({only_orig} only-original, {missing['only_in_refined']} only-refined)")
    for name, t in groups.items():
        print()
        print(f"[{name}] reads compared: {t.reads}")
        if not t.reads:
            continue
        pct = 100.0 * t.cigar_changed / t.reads
        print(f"  CIGAR rewritten        : {t.cigar_changed} ({pct:.1f}%)")
        print(f"  indel structure changed: {t.indel_structure_changed}")
        print(f"  alignment start moved  : {t.start_moved}")
        print(f"  net indel ops          : I {t.d_n_ins:+d}  D {t.d_n_del:+d}")
        print(f"  net indel bases        : I {t.d_ins_bases:+d}  D {t.d_del_bases:+d}")
        print(f"  edit distance NM down/up: {t.nm_decreased}/{t.nm_increased} "
              f"(net {t.d_nm:+d}; not a quality ranking)")
        for ex in t.examples:
            print(f"    {ex['read']}")
            print(f"      pos {ex['ref_start_original']} -> {ex['ref_start_refined']}")
            print(f"      before: {ex['cigar_original'][:110]}")
            print(f"      after : {ex['cigar_refined'][:110]}")

    b, a = loci_before, loci_after
    print()
    print(f"[indel breakpoints across the {len(phased_names)} phased reads] "
          f"before -> after")
    print(f"  distinct indel loci      : {b.distinct_loci} -> {a.distinct_loci} "
          f"({pct_delta(b.distinct_loci, a.distinct_loci)})")
    print(f"  total indel ops          : {b.total_ops} -> {a.total_ops} "
          f"({pct_delta(b.total_ops, a.total_ops)})")
    print(f"  singleton loci (1 read)  : {b.singleton_loci} -> {a.singleton_loci} "
          f"({pct_delta(b.singleton_loci, a.singleton_loci)})")
    print(f"  mean reads per locus     : {b.mean_reads_per_locus:.2f} -> "
          f"{a.mean_reads_per_locus:.2f}")
    print(f"  >={args.min_sv_len}bp indel loci        : {b.sv_distinct_loci} -> "
          f"{a.sv_distinct_loci} "
          f"({pct_delta(b.sv_distinct_loci, a.sv_distinct_loci)})   "
          f"<- consolidation of large indels")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nJSON -> {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
