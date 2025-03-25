#!/usr/bin/env python
# import
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import tiledbsoma
import tiledbsoma.io


# logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

# argparse
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

desc = 'Get the number of cells per gene in a tiledb-soma database.'
epi = """DESCRIPTION:
    Get the number of cells per gene in a tiledb-soma database. The
    output is a csv file with columns: soma_joinid, var_id, cell_count.
    The cell count is the number of cells that have a non-zero value
    for the gene.
    
    Example usage:
    $ python cell-counts-per-gene.py /path/to/db
"""
parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                 formatter_class=CustomFormatter)
parser.add_argument('db_uri', type=str, help='tiledb-soma uri')
parser.add_argument('--outfile', type=str, default='cell_counts.csv', help='Output file name')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size for reading data matrix')


# functions
def get_var_meta(db_uri: str):
    logging.info('Reading in var metadata...')
    # read in the var level data
    with tiledbsoma.Experiment.open(db_uri) as exp:
        df = (
            exp.ms["RNA"]
            .var.read(column_names=["soma_joinid", "var_id"])
            .concat()
            .to_pandas()
        )
    return df

def get_cells_per_var(db_uri: str, start: int, end: int) -> np.ndarray:
    """
    Get the number of cells per gene in the range [start
    end). This is done by reading the data matrix and counting
    the number of cells that have a non-zero value for each gene.
    """
    logging.info(f'Getting cell counts per gene in range [{start}, {end})...')
    with tiledbsoma.Experiment.open(db_uri) as exp:
        data = exp.ms["RNA"].X["data"].read((slice(None), slice(start, end))).coos().concat()
    sp = data.to_scipy().tocsc()
    return np.diff(sp.indptr)[start:end]

def main(args):
    # get var (gene) metadata
    df_var = get_var_meta(args.db_uri)

    # add batch column
    df_var["batch"] = df_var["var_id"] // args.batch_size

    # get cell counts per gene
    df_var["cell_count"] = get_cells_per_var(args.db_uri, 0, df_var.shape[0])

    # write to file
    df_var.to_csv(args.outfile, index=False)
    logging.info(f'Output written to: {args.outfile}')
    
# main
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)