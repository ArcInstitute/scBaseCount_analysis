import os
import math
import tempfile
import subprocess
from typing import List

import submitit
import pandas as pd
import pyarrow.dataset as ds
import gcsfs
import scanpy as sc
import anndata as ad


# -----------------------
# Config
# -----------------------
GCS_BASE_PATH = "gs://arc-scbasecount/2025-02-25/"
FEATURE_TYPE = "GeneFull_Ex50pAS"
TF_CHECKPOINT = "/data/cachris/tf_embeddings/checkpoints/tf_metazoa"
OUT_DIR = "./tf_out2"
N_CHUNKS = 25

ALLOWED_ORGS = [
    "Mus musculus", "Homo sapiens", "Danio rerio", "Oryctolagus cuniculus",
    "Drosophila melanogaster"
]


# -----------------------
# Helpers
# -----------------------
def chunkify(items: List[str], n_chunks: int) -> List[List[str]]:
    items = list(items)
    k = math.ceil(len(items) / n_chunks)
    return [items[i:i+k] for i in range(0, len(items), k)]


def build_embryo_file_list() -> List[str]:
    fs = gcsfs.GCSFileSystem()

    # path to metadata parquet files
    gcs_path = "/".join([GCS_BASE_PATH.rstrip("/"), "metadata", FEATURE_TYPE])

    # find sample_metadata.parquet files
    files = fs.glob("/".join([gcs_path.rstrip("/"), "**"]))
    files = [f for f in files if os.path.basename(f) == "sample_metadata.parquet"]

    embryos = []
    for f in files:
        meta = ds.dataset(f, filesystem=fs, format="parquet").to_table().to_pandas()
        meta = meta[meta["tissue"].str.lower().str.contains("embryo", na=False)]
        embryos.append(meta)

    embryos = pd.concat(embryos, ignore_index=True)
    embryos = embryos[embryos["organism"].isin(ALLOWED_ORGS)]

    # unique h5ad paths to process
    return list(embryos["file_path"].unique())


# -----------------------
# Work function (runs on cluster)
# -----------------------
def process_chunk(file_paths: List[str], chunk_id: int, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fs = gcsfs.GCSFileSystem()

    adatas = []

    for i, infile in enumerate(file_paths):
        print(f"[chunk {chunk_id}] Processing {i+1}/{len(file_paths)}: {infile}")

        with tempfile.TemporaryDirectory() as td:
            local_in = os.path.join(td, "input.h5ad")
            local_out = os.path.join(td, "output.emb.h5ad")

            # copy from GCS
            fs.get(infile, local_in)

            # prep h5ad locally (avoid shared filenames)
            a = sc.read_h5ad(local_in)
            a.var["ensembl_id"] = a.var_names
            a.obs["assay"] = 31
            a.obs['barcode'] = a.obs_names
            a.write_h5ad(local_in)  # overwrite in tempdir

            # transcriptformer inference
            cmd = [
                "transcriptformer", "inference",
                "--checkpoint-path", TF_CHECKPOINT,
                "--data-file", local_in,
                "--output-filename", local_out,
            ]
            subprocess.run(cmd, check=True)

            # read embeddings output and keep small object
            emb = sc.read_h5ad(local_out)
            small = ad.AnnData(X=emb.obsm["embeddings"], obs=a.obs.copy())
            adatas.append(small)

    # write partial output for this chunk
    out_path = os.path.join(out_dir, f"tf_embeddings_chunk{chunk_id:02d}.h5ad")
    merged = ad.concat(adatas, join="outer", merge="same")
    merged.write_h5ad( out_path)

    print(f"[chunk {chunk_id}] Wrote: {out_path}")
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    file_list = build_embryo_file_list()
    print(f"Total unique files: {len(file_list)}")

    chunks = chunkify(file_list, N_CHUNKS)
    print("Chunk sizes:", [len(c) for c in chunks])

    executor = submitit.AutoExecutor(folder=os.path.join(OUT_DIR, "submitit_logs"))
    executor.update_parameters(
        # adjust to your cluster
        slurm_array_parallelism=5,
        slurm_gres="gpu:1",
        slurm_cpus_per_task=15,
        slurm_mem="100G",
        slurm_time="12:00:00",

        # slurm_constraint="a100",  # if needed
        #slurm_additional_parameters={"cpu-bind": "none"},
        #slurm_srun_args=["--cpu-bind=none"],
        )

    jobs = []
    with executor.batch():
        for chunk_id, chunk in enumerate(chunks):
            jobs.append(executor.submit(process_chunk, chunk, chunk_id, OUT_DIR))

    print("Submitted jobs:")
    for j in jobs:
        print(" ", j.job_id)

    # Optionally: write a manifest of expected outputs
    manifest = os.path.join(OUT_DIR, "chunk_jobs_manifest.txt")
    with open(manifest, "w") as f:
        for chunk_id in range(len(chunks)):
            f.write(os.path.join(OUT_DIR, f"tf_embeddings_chunk{chunk_id:02d}.h5ad") + "\n")
    print("Manifest:", manifest)


if __name__ == "__main__":
    main()
