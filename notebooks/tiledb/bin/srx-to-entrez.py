#!/usr/bin/env python
import sys
import logging
import argparse
import requests
import xml.etree.ElementTree as ET
import csv
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


desc = 'Convert SRX accessions to Entrez Gene IDs.'
epi = """DESCRIPTION:
This script converts a list of SRX accessions to Entrez Gene IDs using the NCBI Entrez API.
It supports retries with exponential backoff and parallel execution for faster processing.
"""
parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                 formatter_class=CustomFormatter)
parser.add_argument('csv', type=str, help='CSV file with SRX accessions')
parser.add_argument('--target-col', type=str, default='SRX_accession',
                    help='Column name containing SRX accessions')
parser.add_argument('--outfile', type=str, default='entrez_ids.csv',
                    help='Output file name for Entrez Gene IDs')
parser.add_argument('--max-workers', type=int, default=4,
                    help='Maximum number of parallel workers')
parser.add_argument('--quiet', action='store_true', help='Suppress logging messages')


def get_session() -> requests.Session:
    """
    Creates a persistent requests session with retries and backoff.
    
    Returns:
        requests.Session: Configured session with retry logic.
    """
    session = requests.Session()
    retries = Retry(
        total=10, 
        backoff_factor=2, 
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


# Persistent session for all network requests
session = get_session()

def fetch(url: str) -> bytes:
    """
    Makes a GET request and returns the content of the response.
    
    Args:
        url (str): The URL to fetch.
    
    Returns:
        bytes: Content of the response.
    """
    response = session.get(url)
    response.raise_for_status()
    return response.content


def srx_to_entrez(srx_id: str) -> List[str]:
    """
    Convert an SRX accession to Entrez Gene IDs using the NCBI Entrez API.
    
    Args:
        srx_id (str): The SRX accession number (e.g., "SRX123456").
    
    Returns:
        List[str]: A list of Entrez Gene IDs associated with the SRX accession.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    try:
        # Step 1: Get the BioSample associated with the SRX accession
        sra_search_url = f"{base}esearch.fcgi?db=sra&term={srx_id}&retmode=xml"
        xml1 = ET.fromstring(fetch(sra_search_url))
        ids = xml1.findall(".//Id")
        if not ids:
            logging.warning(f"No results found for {srx_id}")
            return []

        sra_id = ids[0].text

        # Step 2: Find the BioSample linked to this SRA ID
        biosample_search_url = f"{base}elink.fcgi?db=biosample&dbfrom=sra&id={sra_id}&retmode=xml"
        xml2 = ET.fromstring(fetch(biosample_search_url))
        biosample_ids = [x.text for x in xml2.findall(".//Id")]
        if not biosample_ids:
            logging.warning(f"No BioSample found for {srx_id}")
            return []

        biosample_id = biosample_ids[0]

        # Step 3: Retrieve the Entrez Gene ID linked to this BioSample
        gene_search_url = f"{base}elink.fcgi?db=gene&dbfrom=biosample&id={biosample_id}&retmode=xml"
        xml3 = ET.fromstring(fetch(gene_search_url))
        entrez_ids = [x.text for x in xml3.findall(".//Id")]
        
        if not entrez_ids:
            logging.warning(f"No Entrez Gene ID found for {srx_id}")
            return []

        return entrez_ids
    except Exception as e:
        logging.error(f"Error processing {srx_id}: {e}")
        return ["NA"]


def main(args: argparse.Namespace) -> None:
    """
    Main function to read input CSV, process SRX accessions in parallel, 
    and write the Entrez Gene IDs to an output file.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Read the CSV to get the list of SRX accessions
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        srx_ids = [row[args.target_col] for row in reader]

    # Store the results
    results = []

    # Parallel processing of SRX to Entrez conversion
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(srx_to_entrez, srx): srx for srx in srx_ids}
        for future in concurrent.futures.as_completed(future_map):
            srx = future_map[future]
            try:
                for entrez_id in future.result():
                    results.append([srx, entrez_id])
                    if not args.quiet and len(results) % 10 == 0:
                        logging.info(f"Processed {len(results)} SRX accessions")
            except Exception as e:
                logging.error(f"Error with {srx}: {e}")

    # Write results to the output file
    with open(args.outfile, 'w') as out:
        out.write('srx_id,entrez_id\n')
        for srx_id, entrez_id in results:
            out.write(f'{srx_id},{entrez_id}\n')
    
    logging.info(f"Results written to {args.outfile}")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)