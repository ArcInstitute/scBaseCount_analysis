import subprocess
import os
import scanpy as sc
from tqdm import tqdm
import gc

for file in os.listdir('//home/cachris/data/tf_embeddings/fetal_atlas/'):
    if '.h5ad' in file and not 'embedding' in file:

        ad = sc.read_h5ad('/home/cachris/data/tf_embeddings/fetal_atlas/Cerebrum_gene_count.RDS.h5ad')
        ad.var['ensembl_id'] = ad.var_names.str.split('.').str[0]
        ad.obs['assay'] = 26
        ad = ad[:,ad.var['gene_type'] == 'protein_coding'].copy()
        ad.write_h5ad(f'//home/cachris/data/tf_embeddings/fetal_atlas/h5ad2/{file}')
        del ad
        gc.collect()
        subprocess.run(f'transcriptformer inference --oom-dataloader --checkpoint-path /home/cachris/data/tf_embeddings/checkpoints/tf_metazoa \
            --data-file /home/cachris/data/tf_embeddings/fetal_atlas/h5ad2/{file} --batch-size 4 --output-filename {file}_embeddings.h5ad', shell = True)
