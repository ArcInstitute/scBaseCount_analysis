import os
from pathlib import Path
import pandas as pd
import scanpy as sc
import gseapy
import numpy as np
import decoupler as dc
import tempfile
import os
import shutil
import tempfile
import gcsfs
from concurrent.futures import ProcessPoolExecutor, as_completed



# IMPORTANT: define this at top-level for multiprocessing pickling
def read_score(path: str, gmt_path: str, score_jobs: int = 1) -> pd.DataFrame:
    fs = gcsfs.GCSFileSystem()
    net = dc.pp.read_gmt(gmt_path)
    with tempfile.TemporaryDirectory(prefix="h5ad_") as tmpdir:
        local_path = os.path.join(tmpdir, os.path.basename(path))
        with fs.open(path, "rb") as src, open(local_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        adata = sc.read_h5ad(local_path)
    adata.var_names = adata.var['gene_symbols'].astype(str)
    adata.var_names_make_unique()

    # Consider filtering thresholds explicitly if you care about reproducibility
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    genes = adata.obs["n_genes_by_counts"]

    median = np.median(genes)
    std = genes.std()

    lower = max(median - 3 * std, 0)
    upper = median + 3 * std
    lower = np.max([lower, 2000])
    adata = adata[
        (adata.obs["n_genes_by_counts"] >= lower) &
        (adata.obs["n_genes_by_counts"] <= upper)
    ]
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    # Assumes you have score_all_programs defined somewhere
    dc.mt.zscore(adata, net, tmin=3)
    scores_df = adata.obsm['score_zscore']
    # Keep track of provenance
    scores_df.insert(0, "source_file", os.path.basename(path))
    out_dir = Path("msigdb_scores_parts")
    out_dir.mkdir(parents=True, exist_ok=True)

    def safe_stem(path: str) -> str:
            return Path(path).stem.replace(" ", "_")

        # inside your loop / worker result handling:
   
   
    out_dir = './'
    out_path = f"./results2/{safe_stem(path)}.parquet"
    scores_df.to_parquet(out_path, compression="zstd")
    print('written!')
    return scores_df


def main():
    data_dir = Path("gs://arc-ctc-nextflow/scBaseCount-publish/prod/2026-01-12/h5ad/GeneFull_Ex50pAS/Homo_sapiens/")
    gmt_path = "/home/cachris/gmt/c8.all.v2025.1.Hs.symbols.gmt"

    fs = gcsfs.GCSFileSystem()

    # gcsfs wants paths without the gs:// prefix for glob
    gcs_prefix = 'arc-ctc-nextflow/scBaseCount-publish/prod/2026-01-12/h5ad/GeneFull_Ex50pAS/Homo_sapiens'

    paths = sorted("gs://" + p for p in fs.glob(f"{gcs_prefix}/*.h5ad"))

    if not paths:
        raise FileNotFoundError(f"No .h5ad files found in {data_dir}")
    # Outer parallelism across files.
    # If score_all_programs already uses threads/processes via n_jobs,
    # be careful not to oversubscribe your machine.
    outer_workers = min(20, os.cpu_count() or 1)   # tune this
    inner_score_jobs = 1                          # start with 1 to avoid oversubscription

    dfs = []
    outdir = Path('/home/cachris/ctc/projects/scBaseCamp/scRecount/scBaseCount_analysis/msigdb_gene_sets/results2')
    done_srxs = {p.stem for p in outdir.iterdir() if p.is_file()}
    # If your outputs have suffixes like ".parquet.gz", use this instead:
    # done_srxs = {p.name.split(".")[0] for p in outdir.iterdir() if p.is_file()}

    # Filter input paths whose SRX is already done
    paths = [p for p in paths if p.split('/')[-1].split('.')[0] not in done_srxs]
    print(len(paths))
    print('this many paths')
    with ProcessPoolExecutor(max_workers=outer_workers) as ex:
        futures = {
            ex.submit(read_score, p, gmt_path, inner_score_jobs): p
            for p in paths
        }
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                dfs.append(fut.result())
            except Exception as e:
                print(f"[ERROR] {p}: {e}")

    if not dfs:
        raise RuntimeError("No results produced (all jobs failed?).")

#    results = pd.concat(dfs, ignore_index=True)

    # Write parquet with compression
 #   out_path = "msigdb_results.parquet"
  #  results.to_parquet(out_path, index=False, compression="zstd")
   # print(f"Wrote {out_path} with shape {results.shape}")
    

if __name__ == "__main__":
    main()
