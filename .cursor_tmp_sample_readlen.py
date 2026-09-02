import json, sys, statistics as st
import pysam

MANIFEST = "data/hprc/manifests/chr21_HG002_pangenome_windows.bal.json"
CAP = {"ont": 150000, "pacbio_hifi": 150000, "illumina": 150000}
THRESH = [32768, 65536, 100000, 150000, 200000, 300000]

m = json.load(open(MANIFEST))
entries = m["entries"]

# group manifest windows by modality (preserving distinct regions)
by_mod = {}
for e in entries:
    by_mod.setdefault(e["modality"], []).append((e["truth_bam"], e["region"]))

def parse_region(r):
    c, se = r.split(":")
    s, e = se.split("-")
    return c, int(s) - 1, int(e)

def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

results = {}
for mod, wins in by_mod.items():
    # spread windows evenly across the manifest to avoid start-of-file bias
    n = len(wins)
    # take up to 120 windows spread across the list
    max_windows = 120
    if n > max_windows:
        step = n / max_windows
        idx = sorted(set(int(i * step) for i in range(max_windows)))
        sel = [wins[i] for i in idx]
    else:
        sel = wins
    bam_path = sel[0][0]
    bam = pysam.AlignmentFile(bam_path, "rb")
    lengths = []
    windows_used = 0
    cap = CAP[mod]
    for bp, region in sel:
        if len(lengths) >= cap:
            break
        c, s, e = parse_region(region)
        windows_used += 1
        for read in bam.fetch(c, s, e):
            if read.is_secondary or read.is_supplementary:
                continue
            ql = read.query_length
            if ql is None or ql == 0:
                ql = read.infer_read_length() or 0
            if ql > 0:
                lengths.append(ql)
        if len(lengths) >= cap:
            break
    bam.close()
    lengths.sort()
    N = len(lengths)
    stats = {
        "bam": bam_path,
        "windows_sampled": windows_used,
        "n": N,
        "min": lengths[0] if N else 0,
        "median": pct(lengths, 50),
        "mean": (sum(lengths) / N) if N else 0,
        "p90": pct(lengths, 90),
        "p95": pct(lengths, 95),
        "p99": pct(lengths, 99),
        "max": lengths[-1] if N else 0,
        "exceed": {t: (sum(1 for x in lengths if x > t) / N if N else 0) for t in THRESH},
    }
    results[mod] = stats

for mod in ["ont", "pacbio_hifi", "illumina"]:
    if mod not in results:
        continue
    s = results[mod]
    print(f"=== {mod} ===")
    print(f"bam={s['bam']}")
    print(f"windows_sampled={s['windows_sampled']} n_reads={s['n']}")
    print(f"min={s['min']} median={s['median']:.0f} mean={s['mean']:.0f} "
          f"p90={s['p90']:.0f} p95={s['p95']:.0f} p99={s['p99']:.0f} max={s['max']}")
    for t in THRESH:
        print(f"  frac > {t:>7}: {s['exceed'][t]*100:8.4f}%")
    print()

json.dump(results, open(".cursor_tmp_readlen_results.json", "w"), indent=2)
