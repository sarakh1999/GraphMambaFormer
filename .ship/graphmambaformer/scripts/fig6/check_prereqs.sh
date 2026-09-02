#!/usr/bin/env bash
# Quick sanity check before running fig6 baselines.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ok=0
fail() { echo "FAIL: $*"; ok=1; }
pass() { echo "OK:   $*"; }

if [ "$FIG6_NATIVE" = "1" ]; then
  for t in vg bwa samtools bcftools parallel; do
    command -v "$t" >/dev/null && pass "$t ($(command -v "$t"))" || fail "$t"
  done
  [ -x /opt/deepvariant/bin/make_examples ] && pass "DeepVariant binaries" || fail "DeepVariant binaries"
  [ -x "$PLOT_PY" ] && pass "plot interpreter ($PLOT_PY)" || fail "plot interpreter ($PLOT_PY)"
  # Only the hap.py stage still needs a daemon, via the mounted socket.
  if docker info >/dev/null 2>&1; then
    pass "Docker socket reachable (hap.py stage)"
  else
    echo "WARN: no Docker socket — every stage works except eval_happy.sh"
  fi
else
  if docker info >/dev/null 2>&1; then pass "Docker daemon running"; else fail "Docker daemon — start Docker Desktop"; fi
fi
[ -f "$FULL_GBZ" ] && pass "HPRC GBZ ($FULL_GBZ)" || fail "HPRC GBZ — see data/hprc/LINKS.md"
[ -f "$REF_FA" ] && pass "GRCh38 $CHR ($REF_FA)" || echo "WARN: run fetch_reference.sh"
command -v curl >/dev/null && pass "curl" || fail "curl"

exit "$ok"
