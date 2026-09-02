#!/usr/bin/env bash
# Regenerate data/hprc/sample_links.json and SAMPLE_LINKS.md from HPRC CSVs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$ROOT/data/hprc/.cache"
mkdir -p "$TMP"

HIFI_CSV="$TMP/data_hifi_release2_v1.0.index.csv"
ILL_CSV="$TMP/data_illumina_release2_v1.0.index.csv"
ASM_IDX="$TMP/Year1_assemblies_v2_genbank.index"

curl -L --fail -o "$HIFI_CSV" \
  https://raw.githubusercontent.com/human-pangenomics/hprc_intermediate_assembly/main/data_tables/sequencing_data/data_hifi_release2_v1.0.index.csv
curl -L --fail -o "$ILL_CSV" \
  https://raw.githubusercontent.com/human-pangenomics/hprc_intermediate_assembly/main/data_tables/sequencing_data/data_illumina_release2_v1.0.index.csv
curl -L --fail -o "$ASM_IDX" \
  https://raw.githubusercontent.com/human-pangenomics/HPP_Year1_Assemblies/main/assembly_index/Year1_assemblies_v2_genbank.index

python3 - <<PY
import csv, json, re
from pathlib import Path

root = Path("${ROOT}")
tmp = root / "data/hprc/.cache"
year1 = [ln.strip() for ln in (root/"data/hprc/year1_samples.txt").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
graph44 = set(ln.strip() for ln in (root/"data/hprc/graph_samples_44.txt").read_text().splitlines() if ln.strip() and not ln.startswith("#"))
plus = {"HG002","HG005","HG00733","HG01109","HG01243","HG02055","HG02080","HG02109","HG02145","HG02723","HG02818","HG03098","HG03486","HG03492","NA18906","NA19240","NA20129","NA21309"}

def bucket(sample):
    return f"s3://human-pangenomics/working/HPRC_PLUS/{sample}/raw_data" if sample in plus else f"s3://human-pangenomics/working/HPRC/{sample}/raw_data"

hifi = {}
with (tmp/"data_hifi_release2_v1.0.index.csv").open() as fh:
    for row in csv.DictReader(fh):
        hifi.setdefault(row["sample_id"], []).append({
            "filename": row["filename"],
            "path": row["path"],
            "coverage": row["coverage"],
            "deepconsensus_coverage": row["deepconsensus_coverage"],
            "total_gbp": row["total_gpb"],
        })

ill = {}
with (tmp/"data_illumina_release2_v1.0.index.csv").open() as fh:
    for row in csv.DictReader(fh):
        ill[row["sample_id"]] = {
            "filename": row["filename"],
            "path": row["path"],
            "coverage": row["coverage"],
            "total_gbp": row["total_gbp"],
        }

asm = {}
with (tmp/"Year1_assemblies_v2_genbank.index").open() as fh:
    for line in fh:
        line = line.strip()
        if not line or line.startswith("sample "):
            continue
        parts = line.split("\t") if "\t" in line else re.split(r"\s+(?=s3://|gs://)", line)
        if len(parts) < 3:
            continue
        asm[parts[0]] = {"hap1": parts[1], "hap2": parts[2]}

rows = []
for s in year1:
    rows.append({
        "sample_id": s,
        "in_graph_44": s in graph44,
        "hprc_bucket": bucket(s),
        "portal": f"https://humanpangenome.org/data/?sample={s}",
        "raw_data_browser": f"https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=working/{'HPRC_PLUS' if s in plus else 'HPRC'}/{s}/raw_data/",
        "hifi_runs": hifi.get(s, []),
        "illumina": ill.get(s),
        "assembly_hap1": asm.get(s, {}).get("hap1", ""),
        "assembly_hap2": asm.get(s, {}).get("hap2", ""),
    })

(root/"data/hprc/sample_links.json").write_text(json.dumps(rows, indent=2) + "\n")

lines = [
    "# HPRC sample links (Year-1 + graph-44)",
    "",
    "Generated from HPRC release CSVs. Use this for the mentor chr21 benchmark and scaling to ~40 individuals.",
    "",
    "- **First training sample:** `HG00438`",
    "- **Default chr21 eval sample (GIAB truth):** `HG005`",
    "- **Graph training set (~44):** `data/hprc/graph_samples_44.txt`",
    "",
    "Regenerate with: `./scripts/chr21/generate_sample_links.sh`",
    "",
    "## Quick commands",
    "",
    "```bash",
    "./scripts/chr21/fetch_hprc_sample.sh HG00438 links",
    "SAMPLE=HG005 ./scripts/chr21/run_all.sh",
    "SAMPLE=HG00438 SKIP_TRUTH=1 ./scripts/chr21/run_all.sh",
    "```",
    "",
    "## All Year-1 individuals",
    "",
    "| Sample | In graph-44 | Illumina (DeepVariant) | HiFi runs | Raw S3 |",
    "| --- | --- | --- | --- | --- |",
]
for row in rows:
    ill_row = row["illumina"]
    ill_cell = "—"
    if ill_row:
        ill_cell = f"`{ill_row['filename']}` ({ill_row['coverage']}x) — [cram]({ill_row['path']})"
    hifi_cell = "—"
    if row["hifi_runs"]:
        first = row["hifi_runs"][0]
        extra = f" (+{len(row['hifi_runs'])-1} more)" if len(row["hifi_runs"]) > 1 else ""
        hifi_cell = f"`{first['filename']}` ({first['coverage']}x){extra} — [bam]({first['path']})"
    lines.append(
        f"| `{row['sample_id']}` | {'yes' if row['in_graph_44'] else 'no'} | {ill_cell} | {hifi_cell} | [browser]({row['raw_data_browser']}) |"
    )

(root/"data/hprc/SAMPLE_LINKS.md").write_text("\n".join(lines) + "\n")
print(f"updated {len(rows)} samples")
PY

echo "Done:"
echo "  $ROOT/data/hprc/sample_links.json"
echo "  $ROOT/data/hprc/SAMPLE_LINKS.md"
