#!/usr/bin/env python
# import
## batteries
from __future__ import print_function
import os
import re
import sys
import argparse
import logging
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
from sql_db_utils import db_connect, db_upsert

# logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

# argparse
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

desc = 'Summarize STAR summary files from directory structure'
epi = """DESCRIPTION:
Search through a directory structure to find STAR Summary.csv files and
summarize them into a single table. The script recursively searches for
Summary.csv files in the STAR output directory structure and concatenates
them into a single table. The table is then written to a file and upserted
into the database.
"""
parser = argparse.ArgumentParser(
    description=desc, epilog=epi, formatter_class=CustomFormatter
)
parser.add_argument('--samples-table', type=str, default=None,
                    help='Table of data. Required columns: [accession, feature, path]. path must be path to Summary.csv')
parser.add_argument('--input-dir', type=str, default=None,
                    help='Root directory containing STAR output directories')
parser.add_argument('--sample', type=str, default="",
                    help='Sample name (optional)')
parser.add_argument('--outfile', type=str, default="combined.csv",
                    help='Output file')
parser.add_argument('--skip-database', action='store_true',
                    help='Skip updating the scRecounter SQL database')
parser.add_argument('--skip-upload', action='store_true',
                    help='Skip uploading matrix files to GCP bucket')
parser.add_argument('--gcp-bucket', type=str, default="gs://arc-ctc-screcounter/prodC/",
                    help='GCP bucket to upload results to')
parser.add_argument('--tenant', type=str, default="prod",
                    help='Tenant to use')
parser.add_argument('--max-datasets', type=int, default=None,
                    help='Maximum number of datasets to process')
parser.add_argument('--ignore-missing', action='store_true',
                    help='Ignore accessions with missing files/directories and continue processing')

# functions
def find_summary_files(root_dir: str, max_datasets: int = None, ignore_missing: bool = False) -> pd.DataFrame:
    """
    Find all Summary.csv files in the directory structure.
    Args:
        root_dir: Root directory containing STAR output directories
        max_datasets: Maximum number of datasets to process
        ignore_missing: Ignore accessions with missing files/directories and continue processing
    Returns:
        List of tuples: (file_path, srx_accession, feature_type)
    """
    summary_files = []
    root_path = Path(root_dir)
    # check that the root directory exists
    if not root_path.exists():
        if ignore_missing:
            logging.warning(f"Root directory does not exist: {root_path} (ignoring due to --ignore-missing)")
            return pd.DataFrame(columns=["accession", "feature", "path"])
        else:
            logging.error(f"Root directory does not exist: {root_path}")
            sys.exit(1)
    
    # Search for Summary.csv files recursively
    for summary_file in root_path.rglob("Summary.csv"):
        # Extract SRX accession and feature type from path
        path_parts = summary_file.parts
        
        # Find SRX accession (starts with SRX)
        srx_accession = None
        feature_type = None
        
        for i, part in enumerate(path_parts):
            if part.startswith('SRX'):
                srx_accession = part
                # Feature type is the parent directory of Summary.csv
                feature_type = summary_file.parent.name
                break
        
        if srx_accession and feature_type:
            summary_files.append((srx_accession, feature_type, str(summary_file)))
            logging.info(f"Found: {srx_accession}/{feature_type}/Summary.csv")
        else:
            logging.warning(f"Could not extract SRX/feature from path: {summary_file}")

        if max_datasets is not None and len(summary_files) >= max_datasets:
            logging.info(f"Found '--max-datasets={max_datasets}' datasets, stopping search")
            break

    # convert to dataframe
    summary_files = pd.DataFrame(summary_files, columns=["accession", "feature", "path"])
    return summary_files
    
def check_matrix_files_exist(base_dir: str, feature: str, processing: str, ignore_missing: bool = False) -> bool:
    """
    Check if the matrix files exist in the directory.
    Args:
        base_dir: Base directory containing the matrix files
        feature: Feature type (e.g., Velocyto, Gene, etc.)
        processing: Processing type (raw or filtered)
        ignore_missing: If True, return False instead of raising exceptions for missing files
    Returns:
        True if the matrix files exist, False otherwise
    """
    if not os.path.exists(base_dir):
        if ignore_missing:
            logging.warning(f"Directory does not exist: {base_dir} (ignoring due to --ignore-missing)")
            return False
        else:
            raise FileNotFoundError(f"Directory does not exist: {base_dir}")

    if feature == "Velocyto":
        if processing == "raw":
            target_files = [
                "spliced.mtx.gz", "unspliced.mtx.gz", "ambiguous.mtx.gz", 
                "barcodes.tsv.gz", "features.tsv.gz"
            ]
        elif processing == "filtered":
            target_files = [
                "spliced.mtx.gz", "unspliced.mtx.gz", "ambiguous.mtx.gz", 
                "barcodes.tsv.gz", "features.tsv.gz",
                "UniqueAndMult-EM.mtx.gz", "UniqueAndMult-Uniform.mtx.gz"
            ]
        else:
            raise ValueError(f"Invalid processing: {processing}")
    else:
        target_files = [
            "matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz", "UniqueAndMult-EM.mtx.gz", "UniqueAndMult-Uniform.mtx.gz"
        ]
    
    for file in target_files:
        file_path = os.path.join(base_dir, file)
        if not os.path.exists(file_path):
            if ignore_missing:
                logging.warning(f"File does not exist: {file_path} (ignoring due to --ignore-missing)")
                return False
            else:
                raise FileNotFoundError(f"File does not exist: {file_path}")
    return True

def get_matrix_dirs(summary_files: pd.DataFrame, ignore_missing: bool = False) -> pd.DataFrame:
    """
    Find all matrix directories in the directory structure.
    Args:
        summary_files: DataFrame with columns: [accession, feature, path]
        ignore_missing: If True, skip accessions with missing files/directories
    Returns:
        DataFrame with columns: [accession, feature, processing, path]
    """
    matrix_files = []
    for i,row in summary_files.iterrows():
        parts = str(row["path"]).split("/")
        accession = parts[5]
        feature = parts[7]
        # get the raw and filtered directories
        raw_dir = "/" + os.path.join(*parts[:8], "raw")
        filt_dir = "/" + os.path.join(*parts[:8], "filtered")
        if (check_matrix_files_exist(raw_dir, feature, "raw", ignore_missing) and 
            check_matrix_files_exist(filt_dir, feature, "filtered", ignore_missing)):
            matrix_files.append((accession, feature, "raw", raw_dir))
            matrix_files.append((accession, feature, "filtered", filt_dir))
        elif ignore_missing:
            logging.warning(f"Skipping raw/filtered matrix files for {accession}/{feature} due to missing files")
            continue
    
    return pd.DataFrame(matrix_files, columns=["accession", "feature", "processing", "path"])

def upload_to_gcp_bucket(matrix_files: pd.DataFrame, summary_files: pd.DataFrame, gcp_bucket: str) -> str:
    """
    Upload matrix directories and Summary.csv files to GCP bucket with structured naming.
    Args:
        matrix_files: DataFrame with columns: [accession, feature, processing, path]
        summary_files: DataFrame with columns: [accession, feature, path]
        gcp_bucket: GCP bucket URL (e.g., gs://bucket-name/path/)
    Returns:
        Upload batch timestamp string
    """
    # Generate timestamp for this upload batch
    timestamp = datetime.now().strftime("%Y-%m-%d_00-00-00")
    
    # Remove trailing slash from bucket if present
    gcp_bucket = gcp_bucket.rstrip('/')
    
    logging.info(f"Starting upload batch: SCRECOUNTER_{timestamp}")
    
    # Upload Summary.csv files first
    logging.info("Uploading Summary.csv files...")
    for i, row in summary_files.iterrows():
        accession = row["accession"]
        feature = row["feature"]
        local_path = row["path"]
        
        # Construct GCP destination path for Summary.csv
        gcp_destination = f"{gcp_bucket}/SCRECOUNTER_{timestamp}/STAR/{accession}/{feature}/"
        
        logging.info(f"Uploading Summary.csv: {local_path} -> {gcp_destination}")
        
        # Use gsutil to upload Summary.csv file
        try:
            cmd = ["gsutil", "cp", local_path, f"{gcp_destination}Summary.csv"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logging.info(f"Successfully uploaded Summary.csv for {accession}/{feature}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to upload Summary.csv for {accession}/{feature}: {e}")
            logging.error(f"Command output: {e.stdout}")
            logging.error(f"Command error: {e.stderr}")
            continue
        except FileNotFoundError:
            logging.error("gsutil not found. Please install Google Cloud SDK.")
            sys.exit(1)
    
    # Upload matrix directories
    logging.info("Uploading matrix directories...")
    for i, row in matrix_files.iterrows():
        accession = row["accession"]
        feature = row["feature"]
        processing = row["processing"]
        local_path = row["path"]
        
        # Construct GCP destination path for matrix files
        gcp_destination = f"{gcp_bucket}/SCRECOUNTER_{timestamp}/STAR/{accession}/{feature}/{processing}/"
        
        logging.info(f"Uploading matrix files: {local_path} -> {gcp_destination}")
        
        # Use gsutil to upload directory recursively
        try:
            cmd = ["gsutil", "-m", "cp", "-r", f"{local_path}/*", gcp_destination]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logging.info(f"Successfully uploaded matrix files for {accession}/{feature}/{processing}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to upload matrix files for {accession}/{feature}/{processing}: {e}")
            logging.error(f"Command output: {e.stdout}")
            logging.error(f"Command error: {e.stderr}")
            continue
        except FileNotFoundError:
            logging.error("gsutil not found. Please install Google Cloud SDK.")
            sys.exit(1)
    
    logging.info(f"Upload batch SCRECOUNTER_{timestamp} completed")
    return f"SCRECOUNTER_{timestamp}"

def main(args):
    # set pandas display options
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 300)

    # logging database
    if not args.skip_database:
         # set tenant
        os.environ["DYNACONF"] = args.tenant
        logging.info("Using database to store STAR results")
        logging.info(f"Tenant: {args.tenant}")
        if args.tenant == "test":
            os.environ["GCP_SQL_DB_NAME"] = "sragent-test"

    # if summary table provided, read in
    if args.samples_table:
        summary_files = pd.read_csv(args.samples_table)
        req_cols = ["accession", "feature", "path"]
        missing_cols = [col for col in req_cols if col not in summary_files.columns]
        if missing_cols:
            logging.error(f"Missing required columns in samples table: {missing_cols}")
            sys.exit(1)
        if args.max_datasets is not None:
            acc_to_keep = summary_files["accession"].unique().tolist()[:args.max_datasets]
            summary_files = summary_files[summary_files["accession"].isin(acc_to_keep)]
    else:
        # find all Summary.csv files
        logging.info(f"Searching for Summary.csv files in: {args.input_dir}")
        summary_files = find_summary_files(args.input_dir, args.max_datasets, args.ignore_missing)
    
    if summary_files.shape[0] == 0:
        logging.error("No Summary.csv files found!")
        sys.exit(1)
    logging.info(f"Found {len(summary_files)} Summary.csv files")

    # find associated matrix files
    matrix_files = get_matrix_dirs(summary_files, args.ignore_missing)
    logging.info(f"Found {len(matrix_files)} matrix files")
    logging.info(f"Number of accessions: {matrix_files['accession'].nunique()}")

    # read in all summary csv files and concatenate
    logging.info("Reading in Summary.csv files...")
    df = []
    for i,row in summary_files.iterrows():
        # Check if Summary.csv file exists
        if not os.path.exists(row["path"]):
            if args.ignore_missing:
                logging.warning(f"Summary.csv file does not exist: {row['path']} (ignoring due to --ignore-missing)")
                continue
            else:
                logging.error(f"Summary.csv file does not exist: {row['path']}")
                continue
        # read in summary csv file
        try:
            x = pd.read_csv(row["path"], header=None)
            x.columns = ["category", "value"]
            x["feature"] = row["feature"]
            # format category
            for y in ["Gene", "GeneFull", "GeneFull_Ex50pAS", "GeneFull_ExonOverIntron", "Velocyto"]:
                regex = re.compile(f" {y} ")
                x["category"] = x["category"].apply(lambda z: regex.sub(" feature ", z))
            x = x.pivot(index=['feature'], columns='category', values='value').reset_index()
            x["sample"] = row["accession"]
            df.append(x)
        except Exception as e:
            if args.ignore_missing:
                logging.warning(f"Error reading {row['path']}: {e} (ignoring due to --ignore-missing)")
            else:
                logging.error(f"Error reading {row['path']}: {e}")
            continue
    if not df:
        logging.error("No valid Summary.csv files could be read!")
        sys.exit(1)

    # concatenate all dataframes
    df = pd.concat(df, ignore_index=True)
    logging.info(f"Number of rows in the raw table: {df.shape[0]}")

    # filter to accessions (samples) in matrix_files
    df = df[df["sample"].isin(matrix_files["accession"])]
    
    # status
    logging.info(f"Number of rows in the matrix-filtered table: {df.shape[0]}")
    logging.info(f"Number of accessions: {df['sample'].nunique()}")
    
    # format columns: no spaces and lowercase
    df.columns = df.columns.str.replace(r'\W', '_', regex=True).str.lower() 

    # coerce columns to numeric
    for col in df.columns.to_list():
        if col not in ["feature", "sample"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # float columns to integer
    cols_to_convert = ["estimated_number_of_cells", "number_of_reads", "umis_in_cells"]
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = df[col].fillna(0).replace([float('inf'), -float('inf')], 0).astype(int)    

    # status
    logging.info(f"Number of rows after formatting: {df.shape[0]}")
    logging.info(f"Unique SRX accessions (samples) processed: {df['sample'].nunique()}")
    logging.info(f"Feature types processed: {df['feature'].unique().tolist()}")

    # write output file
    df.to_csv(args.outfile, index=False)
    logging.info(f"Results written to: {args.outfile}")

    # upload matrix files to GCP bucket
    if args.skip_upload:
        logging.info("Skipping GCP bucket upload (--skip-upload flag set)")
        return
    else:
        logging.info("Uploading matrix files to GCP bucket...")
        args.gcp_bucket = args.gcp_bucket.rstrip("/")
        upload_batch = upload_to_gcp_bucket(matrix_files, summary_files, args.gcp_bucket)
        logging.info(f"Upload completed for batch: {upload_batch}")

    # upsert results to database
    if not args.skip_database:
        logging.info("Updating screcounter_star_results...")
        with db_connect() as conn:
            db_upsert(df, "screcounter_star_results", conn)
    logging.info("Done!")

## script main
if __name__ == '__main__':
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(override=True)
    main(args)

    