# Imports
import pandas as pd
import scanpy as sc
import os
import logging
from joblib import dump, load
from sklearn.pipeline import Pipeline
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
import time
from tqdm import tqdm
import psutil
import gc

# SRX adata directories
HUMAN_SRX_DIR = '/large_storage/ctc/public/scBasecamp/GeneFull_Ex50pAS/GeneFull_Ex50pAS/Homo_sapiens'

# SRX embedding directories with State
HUMAN_PARQUET = '/large_storage/ctc/userspace/rohankshah/homo_sapiens.parquet'

# Directory for classifiers and output directory for annotated cells
OUTPUT_DIR = '/large_storage/goodarzilab/arshian/state_ct_obs/Homo_sapiens'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

class MetaSearcher:

    def __init__(self):
        self.metadata = pd.read_parquet('/large_storage/ctc/public/scBasecamp/metadata_tsp_annotated_2.parquet')

    def get_metadata(self):
        return self.metadata

    def get_organism(self, srx):
        if srx not in self.metadata['srx_accession'].values:
            return None
        organism = self.metadata[self.metadata['srx_accession'] == srx]['organism'].values[0]
        return organism

    def get_tissue(self, srx):
        if srx not in self.metadata['srx_accession'].values:
            return None
        tissue = self.metadata[self.metadata['srx_accession'] == srx]['tissue_mapped'].values[0]
        return tissue

    def get_tissue_tsp(self, srx):
        if srx not in self.metadata['srx_accession'].values:
            return None
        tissue = self.metadata[self.metadata['srx_accession'] == srx]['tissue_tsp'].values[0]
        return tissue

    def is_human(self, srx):
        if srx not in self.metadata['srx_accession'].values:
            return None
        organism = self.metadata[self.metadata['srx_accession'] == srx]['organism'].values[0]
        return organism == 'Homo sapiens'

    def is_mouse(self, srx):
        if srx not in self.metadata['srx_accession'].values:
            return None
        organism = self.metadata[self.metadata['srx_accession'] == srx]['organism'].values[0]
        return organism == 'Mus musculus'

meta_searcher = MetaSearcher()

# Run pipeline
def annotate_srx(srx: str, embeds_dir: str, classifiers_dir: str, srx_dir: str, output_dir: str):
    """
    Annotates the cell types of the given `srx` using embeddings from `embeds_df` and classifiers in `classifiers_dir` and saves dataframe to the `output_dir` output file
    """
    start_time = time.time()
    metadata = meta_searcher.get_metadata()
    if f"{srx}.parquet" in os.listdir(output_dir):
        logging.info(f"[PID {os.getpid()}] Already processed {srx}, skipping")
        return
    # Retrieving the tissue model
    logging.info(f"[PID {os.getpid()}] Processing {srx}")
    tissue = meta_searcher.get_tissue_tsp(srx)
    tissue_model = get_tissue_model(tissue, classifiers_dir)
    if not tissue_model:
        logging.error(f"[PID {os.getpid()}] No tissue model found for {srx}, tissue: {tissue}")
        return
    logging.info(f"[PID {os.getpid()}] Loaded in model {tissue}")

    # Get SRX h5ad obs to concatenate cell types to
    srx_adata = sc.read_h5ad(srx_dir + '/' + srx + '.h5ad')
    logging.info(f"[PID {os.getpid()}] Read in SRX anndata")

    # Retrieve the corresponding embeddings
    embeds_df = get_embeddings(srx, embeds_dir)
    assert len(embeds_df) == len(srx_adata.obs), "Embeddings and SRX are not the same size"
    assert len(embeds_df.columns) == 2058, f"Extra columns in embeddings ({len(embeds_df.columns)} dimensions instead of 2058)"
    embeddings_arr = embeds_df.loc[srx_adata.obs_names].to_numpy()
    logging.info(f"[PID {os.getpid()}] Extracting embeddings for SRX successfully")

    # Run the model and retrieve cell types
    cell_types = tissue_model.predict(embeddings_arr)
    srx_adata.obs['cell_type'] = cell_types
    logging.info("[PID {os.getpid()}] Annotated predicted cell types")

    # Save the new annotated obs
    save_path = output_dir + f'/{srx}.parquet'
    srx_adata.obs.to_parquet(save_path)
    logging.info(f"[PID {os.getpid()}] Saved annotated SRX anndata obs dataframe to {save_path}")
    
    # Check time
    end_time = time.time()
    logging.info(f"[PID {os.getpid()}] {srx} took: {end_time - start_time:.2f} seconds")

def get_tissue_model(tissue: str, model_dir: str) -> Pipeline:
    """
    Get the classifier of the corresponding tissue from `model_dir` file path
    """
    model_file = model_dir + f'/{tissue}_ref_model_logreg.joblib'
    if os.path.exists(model_file):
        return load(model_file)
    return None

def get_embeddings(srx: str, embeds_dir: str):
    """
    Get the embeddings of the given `srx` from file path `embeds_dir`
    """
    filters = [('dataset', '=', srx)]
    table = pq.read_table(embeds_dir, filters=filters)
    df = table.to_pandas().iloc[:, 0:2059]
    embeds_df = df.set_index('cell')
    return embeds_df

def process_srx_with_cleanup(srx):
    logging.disable(logging.CRITICAL)
    result = annotate_srx(srx, HUMAN_PARQUET, MODELS_FILE_PATH, HUMAN_SRX_DIR, OUTPUT_DIR)

    gc.collect()
    return result

def execute(batch_size=10, max_workers=10):
    human_srxs = os.listdir(HUMAN_SRX_DIR)
    annotated_srxs = os.listdir(OUTPUT_DIR)
    srx_list = [srx_file.split('.')[0] for srx_file in human_srxs if srx_file.replace('.h5ad', '.parquet') not in annotated_srxs]
    
    failed_srxs = []
    successful_srxs = []
    
    for i in tqdm(range(0, len(srx_list), batch_size)):
        batch = srx_list[i:i+batch_size]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_srx = {executor.submit(process_srx_with_cleanup, srx): srx for srx in batch}
            
            for future in as_completed(future_to_srx):
                srx = future_to_srx[future]
                try:
                    future.result()
                    successful_srxs.append(srx)
                except Exception as e:
                    failed_srxs.append((srx, str(e)))
        
    print(f"Successful: {len(successful_srxs)}")
    print(f"Failed: {len(failed_srxs)}")

    if failed_srxs:
        with open('failed_srxs.txt', 'w') as f:
            for srx, error in failed_srxs:
                f.write(f"{srx}: {error}\n")
        print(f"Failed SRXs saved to failed_srxs.txt")

if __name__ == '__main__':
    execute(batch_size=10, max_workers=10)
