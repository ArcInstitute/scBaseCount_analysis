Analysis Code for scBaseCount
=============================

All code associated with the scBaseCount manuscript.

* [scBaseCount manuscript](https://www.biorxiv.org/content/10.1101/2025.02.27.640494v2)
* [Virtual Cell Atlas](https://arcinstitute.org/tools/virtualcellatlas)
* [Arc website](https://arcinstitute.org)

Repository overview
-------------------
This repo contains Jupyter notebooks, small Python CLI tools, and a Nextflow pipeline used for analyzing scBaseCount datasets, deriving summary statistics, and comparing data sources (e.g., CellXGene vs scBaseCamp).

Notebook summary
----------------
- CellXGene: `CellXGene/cell_counts.ipynb`, `umi_counts.ipynb`, `census_summary.ipynb`.
- SRAgent: `SRAgent/summary_stats.ipynb`, `validation.ipynb`, `run_time_stats.ipynb`.
- Estimation and meta‑analysis: `cell_count_estimates/estimate_from_sragent.ipynb`, `cell_statistics_meta_analysis/MetaAnalysis.ipynb`.
- Tissues and labels: `tissues/tissue_annotate*.ipynb`, `tissue_summary.ipynb`.
- Dataset‑specific and comparisons: `Replogle2022/`, `tiledb/scBaseCamp_vs_CellxGene.ipynb`, `misc/species_umap.ipynb`.

Scripts (CLI utilities)
-----------------------
- SRA helpers: `bioproject2srx.py`, `srx-to-project.py`, `srx-to-entrez-id.py`.
- Ontology/labels: `get-disease-ontology.py`, `cluster-ontology.py`, `tissue_ontology_id2label.py`.
- DB utilities: `ChrisData2SQL-DB.py`, `ChrisDataNoSRX2SQL-DB.py`, `sql_db_utils.py`, `db_utils.py`.

Pipeline
--------
- Nextflow H5AD processing: `nextflow/metaq/main.nf` with config in `nextflow/metaq/nextflow.config`.
  Example: `nextflow run nextflow/metaq/main.nf -resume -profile slurm --input_dir <in> --output_dir <out>`


