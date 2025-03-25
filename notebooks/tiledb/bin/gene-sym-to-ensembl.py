#!/usr/bin/env python
# import
import sys
import logging
import argparse
import requests
from time import sleep
import pandas as pd
import tiledbsoma
import tiledbsoma.io


# logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# argparse
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

desc = 'Convert gene symbols to ensembl ids.'
epi = """DESCRIPTION:
This script converts gene symbols to ensembl ids.
Pulling input from a tiledb-soma database.
"""
parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                 formatter_class=CustomFormatter)
parser.add_argument('db_uri', type=str, help='tiledb-soma uri')
parser.add_argument('--var-col', type=str, default='var_id',
                    help='Column with target var ids')
parser.add_argument('--outfile', type=str, default='ensembl_ids.csv',
                    help='Output file name')


# functions
def convert_to_ensembl(genes: list, species: str="human") -> pd.DataFrame:
    """
    Convert gene symbols to ensembl ids.
    Args:
        genes: list of gene symbols
        species: species name (default: human)
    Returns:
        pandas dataframe with ensembl ids
    """
    species_map = {"human": "homo_sapiens", "mouse": "mus_musculus"}
    species = species_map.get(species.lower(), species)

    results = []
    for i,gene in enumerate(genes, 1):
        # status
        if i % 100 == 0:
            logging.info(f"Processed {i} genes...")

        # store already converted
        if gene.startswith("ENSG"):
            results.append([gene, gene])
            continue
    
        # call api with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = f"https://rest.ensembl.org/xrefs/symbol/{species}/{gene}?content-type=application/json"
                response = requests.get(url)
                if response.ok:
                    data = response.json()
                if data:
                    results.append([gene, data[0]["id"]])
                    break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # Last attempt
                    logging.warning(f"Failed to process gene {gene} after {max_retries} attempts: {str(e)}")
                else:
                    logging.info(f"Retry {attempt + 1} for gene {gene}")
            sleep(attempt + 1)

    # convert results to dataframe
    return pd.DataFrame(results, columns=['gene_id', 'ensembl_id'])
 
def load_var_ids(db_uri: str, target_col: str) -> list:
    """
    Load variable ids from tiledb-soma database.
    Args:
        db_uri: tiledb-soma uri
        target_col: column name to extract 
    Returns:
        list of variable ids
    """
    with tiledbsoma.Experiment.open(db_uri) as exp:
        df = (
            exp.ms["RNA"]
            .var.read(column_names=[target_col])
            .concat()
            .to_pandas()
        )
    return df[target_col].tolist()


def main(args):
    # load variable ids
    var_ids = load_var_ids(args.db_uri, target_col=args.var_col)

    # convert
    ensembl_ids = convert_to_ensembl(var_ids)

    # write out
    ensembl_ids.to_csv(args.outfile, index=False, sep=",")
    logging.info(f"Output written to: {args.outfile}")

    
# main
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)