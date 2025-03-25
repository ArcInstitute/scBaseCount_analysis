# import
import sys
import time
import concurrent.futures
from typing import Optional, List, Tuple, Union

import pandas as pd
import polars as pl
import scanpy as sc
import scipy.sparse as sp
import plotnine as pn
import pyarrow as pa
import pyarrow.compute as pc
import anndata
import tiledbsoma
import tiledbsoma.io


# functions
def tiledbsoma_open(db_uri: str, max_retries: int=5):
    """
    Open a TileDB-SOMA database with retries.
    Parameters:
        db_uri: str
            URI of the TileDB-SOMA database.
        max_retries: int
            Maximum number of retries.
    Returns:
        tiledbsoma.Array
            Opened TileDB-SOMA database.
    """
    for i in range(max_retries):
        try:
            return tiledbsoma.open(db_uri)
        except tiledbsoma.SOMAError:
            print("Waiting for the database to become available...")
            time.sleep(i+1)
    raise RuntimeError("Failed to open database after multiple retries.")

def tiledbsoma_exp_open(db_uri: str, max_retries: int=5):
    """
    Open a TileDB-SOMA database with retries.
    Parameters:
        db_uri: str
            URI of the TileDB-SOMA database.
        max_retries: int
            Maximum number of retries.
    Returns:
        tiledbsoma.Array
            Opened TileDB-SOMA database.
    """
    for i in range(max_retries):
        try:
            return tiledbsoma.Experiment.open(db_uri)
        except tiledbsoma.SOMAError:
            print("Waiting for the database to become available...")
            time.sleep(i+1)
    raise RuntimeError("Failed to open database after multiple retries.")

def get_obs_OLD(
        db_uri: str,
        columns: Optional[List[str]] = None,
        obs_query: Optional[tiledbsoma.AxisQuery] = None,
        group_by: Optional[List[str]] = None,
        axis: str = "RNA",
        agg_name: str = "count_all",
        max_tries: int = 5,
    ) -> pd.DataFrame:
    """
    Retrieve the obs table from a tiledbsoma Experiment.
    If group_by is provided, aggregated counts by the specified grouping columns are returned.
    Otherwise, the full obs subset with the specified columns is returned.
    Parameters:
      db_uri: URI for the database.
      columns: List of column names to retrieve. All columns used if not provided and no group_by.
      obs_query: Optional AxisQuery to filter the obs data.
      group_by: Optional list of columns to group by for aggregation.
      axis: The axis to query (default "RNA").
      agg_name: Name for the aggregated count column (default "count_all").
      max_tries: Maximum number of retries for reading data.
    Returns:
      A pandas DataFrame containing either the subset of obs data or the aggregated counts.
    """
    if group_by is not None:
        # When grouping, we only need the group_by columns.
        group_cols = group_by
        chunk_results: List[pd.DataFrame] = []
        with tiledbsoma.Experiment.open(db_uri) as exp:
            # Use axis_query if provided.
            if obs_query is not None:
                for i in range(max_tries):
                    try:
                        reader = exp.axis_query(axis, obs_query=obs_query).obs(column_names=group_cols)
                        break
                    except TypeError:
                        reader = exp.axis_query(axis, obs_query=obs_query).obs()
                        break
                    except tiledbsoma.SOMAError:
                        print(f"Failed to read obs data on try {i+1}. Retrying...")
                        time.sleep(i+2)
                chunk_iter = reader
            else:
                chunk_iter = exp.obs.read(column_names=group_cols)
            # Process each chunk.
            for chunk in chunk_iter:
                if chunk.num_rows == 0:
                    continue
                df_chunk = chunk.to_pandas()
                # Group and count within this chunk.
                df_group = df_chunk.groupby(group_cols, as_index=False).size().rename(columns={"size": agg_name})
                chunk_results.append(df_group)
        if not chunk_results:
            return pd.DataFrame(columns=group_cols + [agg_name])
        # Combine chunk results and re-aggregate to get overall counts.
        df_all = pd.concat(chunk_results, ignore_index=True)
        df_final = df_all.groupby(group_cols, as_index=False)[agg_name].sum()
        return df_final.sort_values(by=agg_name, ascending=False)
    else:
        # Without grouping, ensure columns is provided.
        if columns is None:
            with tiledbsoma_exp_open(db_uri) as exp:
                columns = [x.name for x in exp.obs.schema]
        chunks: List[pd.DataFrame] = []
        with tiledbsoma_exp_open(db_uri) as exp:
            if obs_query is not None:
                for i in range(max_tries):
                    try:
                        reader = exp.axis_query(axis, obs_query=obs_query).obs(column_names=columns)
                        break
                    except TypeError:
                        reader = exp.axis_query(axis, obs_query=obs_query).obs()
                        break
                    except tiledbsoma.SOMAError:
                        print(f"Failed to read obs data on try {i+1}. Retrying...")
                        time.sleep(i+2)
                chunk_iter = reader
            else:
                chunk_iter = exp.obs.read(column_names=columns)
            for chunk in chunk_iter:
                if chunk.num_rows == 0:
                    continue
                df_chunk = chunk.to_pandas()[columns]
                chunks.append(df_chunk)
        if not chunks:
            return pd.DataFrame(columns=columns)
        return pd.concat(chunks, ignore_index=True)

def get_obs(
    db_uri: str,
    columns: Optional[List[str]] = None,
    obs_query: Optional[tiledbsoma.AxisQuery] = None,
    group_by: Optional[List[str]] = None,
    axis: str = "RNA",
    agg_name: str = "count_all",
    max_tries: int = 5,
    return_polars: bool = False
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Retrieve the obs table from a tiledbsoma Experiment.
    
    If group_by is provided, aggregated counts by the specified grouping columns are returned.
    Otherwise, the full obs subset with the specified columns is returned.
    
    Parameters:
      db_uri: URI for the database.
      columns: List of column names to retrieve. If None and group_by is not used, all columns are used.
      obs_query: Optional AxisQuery to filter the obs data.
      group_by: Optional list of columns to group by for aggregation.
      axis: The axis to query (default "RNA").
      agg_name: Name for the aggregated count column (default "count_all").
      max_tries: Maximum number of attempts to read the data.
      return_polars: If True, returns a Polars DataFrame; otherwise, returns a Pandas DataFrame.
    
    Returns:
      A DataFrame containing either the subset of obs data or the aggregated counts, in the desired format.
    """
    # Helper: convert a chunk to the desired DataFrame type using Arrow as intermediary
    def convert_chunk(chunk, cols: List[str]) -> Union[pd.DataFrame, pl.DataFrame]:
        if return_polars:
            try:
                arrow_table = chunk.to_arrow()
            except AttributeError:
                arrow_table = pa.Table.from_pandas(chunk.to_pandas())
            return pl.from_arrow(arrow_table)[cols]
        else:
            return chunk.to_pandas()[cols]
    
    if group_by is not None:
        group_cols = group_by
        chunk_results = []  # type: List[Union[pd.DataFrame, pl.DataFrame]]
        with tiledbsoma.Experiment.open(db_uri) as exp:
            # Use axis_query if provided.
            if obs_query is not None:
                for i in range(max_tries):
                    try:
                        reader = exp.axis_query(axis, obs_query=obs_query).obs(column_names=group_cols)
                        break
                    except TypeError:
                        reader = exp.axis_query(axis, obs_query=obs_query).obs()
                        break
                    except tiledbsoma.SOMAError:
                        print(f"Failed to read obs data on try {i + 1}. Retrying...")
                        time.sleep(i + 2)
                chunk_iter = reader
            else:
                chunk_iter = exp.obs.read(column_names=group_cols)
            # Process each chunk.
            for chunk in chunk_iter:
                if chunk.num_rows == 0:
                    continue
                if return_polars:
                    try:
                        arrow_table = chunk.to_arrow()
                    except AttributeError:
                        arrow_table = pa.Table.from_pandas(chunk.to_pandas())
                    df_chunk = pl.from_arrow(arrow_table)
                    # Group and count rows using Polars.
                    df_group = df_chunk.groupby(group_cols).agg(pl.count().alias(agg_name))
                else:
                    df_chunk = chunk.to_pandas()
                    df_group = df_chunk.groupby(group_cols, as_index=False).size().rename(columns={"size": agg_name})
                chunk_results.append(df_group)
        if not chunk_results:
            if return_polars:
                result_df = pl.DataFrame({col: [] for col in group_cols + [agg_name]})
            else:
                result_df = pd.DataFrame(columns=group_cols + [agg_name])
        else:
            if return_polars:
                df_all = pl.concat(chunk_results)
                result_df = df_all.groupby(group_cols).agg(pl.col(agg_name).sum()).sort(agg_name, reverse=True)
            else:
                df_all = pd.concat(chunk_results, ignore_index=True)
                result_df = df_all.groupby(group_cols, as_index=False)[agg_name].sum().sort_values(by=agg_name, ascending=False)
    else:
        if columns is None:
            with tiledbsoma_exp_open(db_uri) as exp:
                columns = [x.name for x in exp.obs.schema]
        chunks = []  # type: List[Union[pd.DataFrame, pl.DataFrame]]
        with tiledbsoma_exp_open(db_uri) as exp:
            if obs_query is not None:
                for i in range(max_tries):
                    try:
                        reader = exp.axis_query(axis, obs_query=obs_query).obs(column_names=columns)
                        break
                    except TypeError:
                        reader = exp.axis_query(axis, obs_query=obs_query).obs()
                        break
                    except tiledbsoma.SOMAError:
                        print(f"Failed to read obs data on try {i + 1}. Retrying...")
                        time.sleep(i + 2)
                chunk_iter = reader
            else:
                chunk_iter = exp.obs.read(column_names=columns)
            for chunk in chunk_iter:
                if chunk.num_rows == 0:
                    continue
                chunks.append(convert_chunk(chunk, columns))
        if not chunks:
            if return_polars:
                result_df = pl.DataFrame({col: [] for col in columns})
            else:
                result_df = pd.DataFrame(columns=columns)
        else:
            if return_polars:
                result_df = pl.concat(chunks)
            else:
                result_df = pd.concat(chunks, ignore_index=True)
    return result_df

def group_contiguous(sorted_ids: List[int]) -> List[Tuple[int, int, List[int]]]:
    """
    Group a sorted list of integers into contiguous blocks.
    Each block is returned as a tuple (start, end, group_ids),
    where group_ids is the list of consecutive integers, and the block
    covers indices from start to end-1.
    Parameters:
        sorted_ids: A sorted list of integers.
    Returns:
        List of tuples (start, end, group_ids) representing contiguous blocks.
    """
    groups = []
    if not sorted_ids:
        return groups
    current = [sorted_ids[0]]
    for i in sorted_ids[1:]:
        if i == current[-1] + 1:
            current.append(i)
        else:
            groups.append((current[0], current[-1] + 1, current.copy()))
            current = [i]
    groups.append((current[0], current[-1] + 1, current.copy()))
    return groups

def read_block(ms_array, block_range: Tuple[int, int], group_ids: List[int]) -> sp.csr_matrix:
    """
    Read a contiguous slice of the measurement array and extract only the rows
    corresponding to group_ids.
    The TileDB array returns a full matrix with data in the original row positions,
    so we use the original group_ids for indexing rather than computing relative indices.
    Parameters:
        ms_array: The TileDB-SOMA array object.
        block_range: Tuple of (start, end) indices for the block.
        group_ids: List of row indices to extract.
    Returns:
        sp.csr_matrix: The sparse matrix with only the selected rows.
    """
    start, end = block_range
    
    # Read and convert to sparse matrix
    block_tensor = ms_array.read(coords=(slice(start, end), slice(None)))
    block_sparse = block_tensor.coos().concat().to_scipy().tocsr()
    
    # Extract rows using original group_ids
    return block_sparse[group_ids, :]

def get_anndata(
    db_uri: str,
    obs_query: Optional[tiledbsoma.AxisQuery] = None,
    measurement_name: str = "RNA",
    X_name: str = "X",
    obs_columns: Optional[List[str]] = None,
    max_workers: int = 1,
    max_tries: int = 5,
) -> anndata.AnnData:
    """
    Retrieve a subset of an experiment as an AnnData object, filtering the measurement
    (X) on-disk by reading only the rows corresponding to the obs query's 'soma_joinid' values.
    In addition, the function reads the var DataFrame from the measurement so that
    the returned AnnData object contains full feature metadata.
    Steps:
      1. Read obs metadata (optionally in chunks) and filter it via obs_query.
      2. Extract the 'soma_joinid' values from obs and group them into contiguous blocks.
      3. Use a ThreadPoolExecutor to read each contiguous block (via slicing) concurrently.
      4. Vertically stack the resulting blocks and reorder rows to match the original obs order.
      5. Read the var DataFrame from the measurement.
      6. Construct an AnnData object with obs, var, and the sparse X matrix.
    Parameters:
      db_uri: URI of the TileDB-SOMA experiment.
      obs_query: Query to filter observations.
      measurement_name: Name of the measurement (e.g. "RNA").
      X_name: Name of the measurement matrix (e.g. "X").
      obs_columns: List of columns to retrieve from obs.
      chunk_obs: Whether to read obs metadata in chunks.
      max_workers: Maximum number of threads for parallel block reads.
      max_tries: Maximum number of retries for reading a block.
    Returns:
      anndata.AnnData: The filtered AnnData object with a sparse X and complete var metadata.
    """
    print("Reading obs metadata...", file=sys.stderr)
    # --- Read obs metadata ---
    if obs_columns is None:
        obs_columns = ["soma_joinid"]
    if "soma_joinid" not in obs_columns:
        obs_columns.append("soma_joinid")
    obs_df = get_obs(db_uri, columns=obs_columns, obs_query=obs_query, axis=measurement_name)
    
    # --- Ensure 'soma_joinid' exists and extract join IDs ---
    if "soma_joinid" not in obs_df.columns:
        raise ValueError("The obs metadata must include the 'soma_joinid' column.")
    required_ids = obs_df["soma_joinid"].tolist()
    sorted_ids = sorted(required_ids)
    groups = group_contiguous(sorted_ids)
    print(f"Found {len(groups)} contiguous blocks.", file=sys.stderr)
    
    print("Reading measurement data by contiguous blocks...", file=sys.stderr)
    block_list = []
    with tiledbsoma_exp_open(db_uri) as exp:
        ms_obj = exp.ms[measurement_name]
        if X_name not in ms_obj:
            available = list(ms_obj.keys())
            raise ValueError(f"Measurement '{measurement_name}' does not contain layer '{X_name}'. "
                             f"Available layers: {available}")
        ms_layer = ms_obj[X_name]
        if isinstance(ms_layer, tiledbsoma.Collection):
            keys = list(ms_layer.keys())
            for i in range(max_tries):
                try:
                    ms_array = ms_layer["data"] if "data" in keys else ms_layer[keys[0]]
                    break
                except tiledbsoma.SOMAError:
                    print(f"Failed to read layer '{X_name}', retrying ({i+1}/{max_tries})...", file=sys.stderr)
                    time.sleep(i+2)
        else:
            ms_array = ms_layer

        status_div = 10 if len(groups) < 1000 else 100
        
        if max_workers < 2:
            # Process blocks sequentially
            for start, end, group_ids in groups:
                try:
                    block_matrix = read_block(ms_array, (start, end), group_ids)
                    block_list.append((group_ids, block_matrix))
                    if len(block_list) % status_div == 0:
                        print(f"  Processed {len(block_list)} blocks.", file=sys.stderr)
                except Exception as e:
                    raise ValueError(f"Error reading block {start} to {end}") from e
        else:
            # Process blocks concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_block = {
                    executor.submit(read_block, ms_array, (start, end), group_ids): (start, end, group_ids)
                    for (start, end, group_ids) in groups
                }
                for future in concurrent.futures.as_completed(future_to_block):
                    block_info = future_to_block[future]
                    try:
                        block_matrix = future.result()
                        block_list.append((block_info[2], block_matrix))
                        if len(block_list) % status_div == 0:
                            print(f"  Processed {len(block_list)} blocks.", file=sys.stderr)
                    except Exception as e:
                        raise ValueError(f"Error reading block {block_info[0]} to {block_info[1]}") from e
    
    if not block_list:
        raise ValueError("No measurement data was read.")
    
    print("Stacking blocks...", file=sys.stderr)
    block_list.sort(key=lambda x: x[0][0])  
    blocks = [blk for (_, blk) in block_list]
    # Now blocks are (cells x genes), verify dimensions match
    n_genes = blocks[0].shape[1]  # Check number of genes (columns) instead
    for i, block in enumerate(blocks):
        if block.shape[1] != n_genes:
            raise ValueError(f"Block {i} has {block.shape[1]} genes but expected {n_genes}")
    # Stack vertically since blocks are already (cells x genes)
    X_filtered = sp.vstack(blocks).tocsr() 
    
    print("Reading var metadata...", file=sys.stderr)
    with tiledbsoma_exp_open(db_uri) as exp:
        var_reader = exp.ms[measurement_name].var.read()
        var_df = var_reader.concat().to_pandas()
    
    print("Creating AnnData object...", file=sys.stderr)
    adata = anndata.AnnData(X_filtered, obs=obs_df, var=var_df)
    return adata
