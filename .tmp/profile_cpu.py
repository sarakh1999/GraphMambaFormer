"""Profile the classical (CPU) alignment stages to find the real hotspots."""
import cProfile, pstats, io, time, random, sys

from graphmambaformer.config import PipelineConfig, AccelConfig
from graphmambaformer.accel import AccelContext
from graphmambaformer.alignment.pipeline import build_pipeline
from graphmambaformer.data.synthetic import (
    build_reference_from_sequence,
    simulate_reads_from_sequence,
)

REF_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
N_READS = int(sys.argv[2]) if len(sys.argv) > 2 else 24

rng = random.Random(0)
ref_seq = "".join(rng.choice("ACGT") for _ in range(REF_LEN))
reads = simulate_reads_from_sequence(ref_seq, "ont", N_READS, seed=1)
read_seqs = [r.seq for r in reads]
print(f"ref_len={REF_LEN} n_reads={N_READS} "
      f"read_len~{sum(len(s) for s in read_seqs)//len(read_seqs)}", flush=True)

# Force the portable CPU path (single worker so cProfile attributes cost cleanly).
cfg = PipelineConfig(mode="fast")
cfg.accel = AccelConfig(device="cpu", stage_parallel=False, num_workers=1)
accel = AccelContext(cfg.accel, device="cpu")
pipe = build_pipeline(cfg, accel=accel)

t0 = time.time()
ref = pipe.build_reference(ref_seq)
print(f"build_reference: {time.time()-t0:.2f}s", flush=True)

# Warm once (JIT-free, but warms caches), then profile.
pipe.align(read_seqs[:2], ref)

pr = cProfile.Profile()
t0 = time.time()
pr.enable()
results, stats = pipe.align(read_seqs, ref)
pr.disable()
wall = time.time() - t0
print(f"align wall: {wall:.2f}s  ({N_READS/wall:.1f} reads/s)  {stats.summary()}", flush=True)

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(25)
print(s.getvalue())
