# HPRC sample / read links

## What “~40 samples” means

The Year-1 pangenome was built from **~44 individuals** (47 Year-1 assemblies
minus held-out `HG002`, `HG005`, `NA19240`). See `graph_samples_44.txt`.

- **First training sample (HPRC core):** `HG00438`
- **First eval sample (this chr21 pipeline default):** `HG005` (GIAB truth)
- **All resolved read links:** `SAMPLE_LINKS.md` (+ `sample_links.json`)

## Portal / manifests

| What | Link |
| --- | --- |
| HPRC data portal | https://humanpangenome.org/data/ |
| Year-1 assemblies (haplotype FASTA index) | https://github.com/human-pangenomics/HPP_Year1_Assemblies |
| Assembly index (the table with `hap1_aws_fasta`) | https://github.com/human-pangenomics/HPP_Year1_Assemblies/blob/main/assembly_index/Year1_assemblies_v2_genbank.index |
| Pangenome / Giraffe indexes | https://github.com/human-pangenomics/hpp_pangenome_resources |
| Raw-data S3 browser | https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=working/ |
| Illumina index CSV | https://github.com/human-pangenomics/hprc_intermediate_assembly/blob/main/data_tables/sequencing_data/data_illumina_release2_v1.0.index.csv |
| PacBio HiFi index CSV | https://github.com/human-pangenomics/hprc_intermediate_assembly/blob/main/data_tables/sequencing_data/data_hifi_release2_v1.0.index.csv |
| ONT index CSV | https://github.com/human-pangenomics/hprc_intermediate_assembly/blob/main/data_tables/sequencing_data/data_ont_release2_v1.0.index.csv |

## First training sample paths

```bash
# resolved Illumina + HiFi URLs for chr21 pipeline
./scripts/chr21/fetch_hprc_sample.sh HG00438 links

# list raw data for HG00438
aws s3 ls --no-sign-request s3://human-pangenomics/working/HPRC/HG00438/raw_data/

# assemblies (do NOT use these for Giraffe/DeepVariant/Sniffles)
aws s3 ls --no-sign-request \
  s3://human-pangenomics/working/HPRC/HG00438/assemblies/year1_f1_assembly_v2_genbank/
```

See also `SAMPLE_LINKS.md` for all 47 Year-1 individuals.

For **DeepVariant** you need Illumina short reads.  
For **Sniffles** you need long reads (PacBio HiFi or ONT), not Illumina.

## Sniffles

https://github.com/fritzsedlazeck/Sniffles
