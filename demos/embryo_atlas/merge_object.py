import scanpy as sc
import pandas as pd
import numpy as np
import os
import anndata

adata = anndata.concat([sc.read_h5ad(f'/home/cachris/data/tf_embeddings/tf_out2/{x}')   for x in os.listdir('/home/cachris/data/tf_embeddings/tf_out2') if '.h5ad' in x])
# path to tf-embedded fetal atlas from '
refs = os.listdir('/home/cachris/data/tf_embeddings/fetal_atlas/')
refs = [x for x in refs if 'embedding' in x]
ref_ads = []
for path in refs:
    ad = sc.read_h5ad(f'/home/cachris/data/tf_embeddings/fetal_atlas/{path}', backed = 'r')
    # only keep the embeddings to save memory
    ad = anndata.AnnData(X = ad.obsm['embeddings'], obs = ad.obs)
    ref_ads.append(ad)
reference = anndata.concat(ref_ads)
for col in reference.obs.columns:
    adata.obs[col] = 'na'

reference.obs['SRX_accession'] = 'na'
reference.obs['barcode'] = 'na'
reference.obs['organism'] = 'na'
adata = anndata.concat([adata, reference])
sc.pp.pca(adata)
#take PCA to further reduce data size
adata2 = anndata.AnnData(X = (adata.obsm['X_pca']), obs = adata.obs)

for col in adata2.obs.columns:
    if adata2.obs[col].dtype == "object":
        adata2.obs[col] = adata2.obs[col].astype(str)

adata2.write_h5ad('comb_integrated.h5ad')