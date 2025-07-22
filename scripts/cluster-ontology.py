#!/usr/bin/env python3
"""
MONDO Disease Ontology Clustering Tool

Clusters MONDO disease ontology nodes into N clusters weighted by node abundance.
"""

import argparse
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import pronto
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MondoOntologyClusterer:
    """
    Clusters MONDO disease ontology with abundance weighting.
    Abundance data is a table with a column for node_id and a column for node_abundance.
    The graph is a directed graph with nodes representing ontology terms and edges representing is_a relationships.
    The graph is weighted by the geometric mean of the node abundances.
    The clustering is performed using spectral clustering.
    The results are saved to a file.
    """
    
    def __init__(self, ontology_path: str, abundance_path: str):
        """Initialize the clusterer.
        
        Args:
            ontology_path: Path to MONDO OWL file
            abundance_path: Path to abundance table (CSV/TSV with node_name, node_abundance columns)
        """
        self.ontology_path = Path(ontology_path)
        self.abundance_path = Path(abundance_path)
        self.ontology = None
        self.graph = None
        self.pruned_graph = None
        self.abundance_data = None
        self.node_weights = {}
        self.nodes_with_abundance = set()
        
    def load_ontology(self) -> nx.DiGraph:
        """Load MONDO ontology from OWL file and convert to NetworkX graph.
        
        Returns:
            NetworkX directed graph representation of the ontology
        """
        logger.info(f"Loading MONDO ontology from {self.ontology_path}")
        
        try:
            self.ontology = pronto.Ontology(str(self.ontology_path))
            logger.info(f"Loaded ontology with {len(self.ontology)} terms")
        except Exception as e:
            logger.error(f"Error loading ontology: {e}")
            sys.exit(1)
            
        # Create NetworkX graph
        self.graph = nx.DiGraph()
        
        # Add nodes with attributes
        for term in self.ontology.terms():
            self.graph.add_node(
                term.id,
                name=term.name,
                definition=str(term.definition) if term.definition else "",
                obsolete=term.obsolete
            )
        
        # Add edges (relationships)
        for term in self.ontology.terms():
            for parent in term.superclasses(distance=1):
                if parent.id != term.id:  # Avoid self-loops
                    self.graph.add_edge(term.id, parent.id, relationship="is_a")
        
        logger.info(f"Created graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def load_abundance_data(self) -> pd.DataFrame:
        """Load node abundance data.
        
        Returns:
            DataFrame with node names and abundances
        """
        logger.info(f"Loading abundance data from {self.abundance_path}")
        
        try:
            # Try to determine separator
            if self.abundance_path.suffix.lower() == '.csv':
                separator = ','
            else:
                separator = '\t'
                
            self.abundance_data = pd.read_csv(self.abundance_path, sep=separator)
            
            # Ensure required columns exist
            required_cols = ['node_id', 'node_abundance']
            missing_cols = [col for col in required_cols if col not in self.abundance_data.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.error(f"Available columns: {list(self.abundance_data.columns)}")
                sys.exit(1)
                
            logger.info(f"Loaded abundance data for {len(self.abundance_data)} nodes")
            return self.abundance_data
            
        except Exception as e:
            logger.error(f"Error loading abundance data: {e}")
            sys.exit(1)
    
    def create_node_weights(self) -> Dict[str, float]:
        """Create node weight dictionary from abundance data.
        
        Returns:
            Dictionary mapping node IDs to weights
        """
        logger.info("Creating node weights from abundance data")
        
        # Create weights dictionary and track nodes with abundance
        self.node_weights = {}
        self.nodes_with_abundance = set()
        matched_nodes = 0
        
        for _, row in self.abundance_data.iterrows():
            abundance = float(row['node_abundance'])        
            if row['node_id'] in self.graph.nodes():
                self.node_weights[row['node_id']] = abundance
                self.nodes_with_abundance.add(row['node_id'])
                matched_nodes += 1
        
        # Set default weight for unmatched nodes
        default_weight = 1.0
        for node_id in self.graph.nodes():
            if node_id not in self.node_weights:
                self.node_weights[node_id] = default_weight
        
        logger.info(f"Matched {matched_nodes}/{len(self.abundance_data)} abundance entries to ontology nodes")
        logger.info(f"Set default weight {default_weight} for {len(self.graph.nodes()) - matched_nodes} unmatched nodes")
        
        return self.node_weights
    
    def prune_graph(self) -> nx.DiGraph:
        """Remove nodes without abundance values and create direct edges to maintain connectivity.
        
        Returns:
            Pruned graph containing only nodes with abundance values
        """
        logger.info("Pruning graph to keep only nodes with abundance values")
        
        # Create new graph with only nodes that have abundance
        self.pruned_graph = nx.DiGraph()
        
        # Add all nodes with abundance to the new graph
        for node in self.nodes_with_abundance:
            if node in self.graph.nodes():
                self.pruned_graph.add_node(
                    node,
                    **self.graph.nodes[node]
                )
        
        logger.info(f"Added {len(self.pruned_graph.nodes())} nodes with abundance values")
        
        # For each node with abundance, find all other nodes with abundance it can reach
        # and create direct edges
        nodes_to_process = list(self.nodes_with_abundance)
        total_pairs = len(nodes_to_process) * (len(nodes_to_process) - 1) // 2
        processed = 0
        
        logger.info(f"Finding connectivity between {len(nodes_to_process)} nodes...")
        
        for i, source in enumerate(nodes_to_process):
            if source not in self.graph.nodes():
                continue
                
            # Find all nodes with abundance reachable from this source
            for target in nodes_to_process[i+1:]:
                if target not in self.graph.nodes() or source == target:
                    continue
                
                # Check if there's a path from source to target or target to source
                path_exists = False
                path_info = None
                
                try:
                    if nx.has_path(self.graph, source, target):
                        path = nx.shortest_path(self.graph, source, target)
                        path_exists = True
                        path_info = ('forward', path)
                    elif nx.has_path(self.graph, target, source):
                        path = nx.shortest_path(self.graph, target, source)
                        path_exists = True
                        path_info = ('reverse', path)
                except:
                    pass
                
                if path_exists and path_info:
                    direction, path = path_info
                    # Create edge in the pruned graph
                    if direction == 'forward':
                        self.pruned_graph.add_edge(source, target, 
                                                 relationship="is_a_transitive",
                                                 path_length=len(path)-1)
                    else:
                        self.pruned_graph.add_edge(target, source,
                                                 relationship="is_a_transitive", 
                                                 path_length=len(path)-1)
                
                processed += 1
                if processed % 10000 == 0:
                    logger.info(f"Processed {processed}/{total_pairs} node pairs...")
        
        # Also preserve direct edges between nodes with abundance
        original_edges = 0
        for u, v in self.graph.edges():
            if u in self.nodes_with_abundance and v in self.nodes_with_abundance:
                self.pruned_graph.add_edge(u, v, relationship="is_a")
                original_edges += 1
        
        logger.info(f"Pruned graph has {self.pruned_graph.number_of_nodes()} nodes and {self.pruned_graph.number_of_edges()} edges")
        logger.info(f"Preserved {original_edges} original direct edges")
        logger.info(f"Created {self.pruned_graph.number_of_edges() - original_edges} transitive edges")
        
        return self.pruned_graph
    
    def create_weighted_adjacency_matrix(self, use_pruned: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Create weighted adjacency matrix for clustering.
        
        Args:
            use_pruned: Whether to use the pruned graph or original graph
            
        Returns:
            Weighted adjacency matrix as numpy array and node list
        """
        logger.info("Creating weighted adjacency matrix")
        
        # Use pruned or original graph
        graph = self.pruned_graph if use_pruned and self.pruned_graph is not None else self.graph
        
        # Get largest connected component for clustering
        if graph.is_directed():
            largest_cc = max(nx.weakly_connected_components(graph), key=len)
        else:
            largest_cc = max(nx.connected_components(graph), key=len)
        
        subgraph = graph.subgraph(largest_cc)
        
        # Create node list and adjacency matrix
        nodes = list(subgraph.nodes())
        n_nodes = len(nodes)
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Initialize adjacency matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        # Fill adjacency matrix with weights
        for u, v in subgraph.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            # Weight by geometric mean of node abundances
            weight = np.sqrt(self.node_weights[u] * self.node_weights[v])
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight  # Make symmetric for clustering
        
        logger.info(f"Created {adj_matrix.shape} weighted adjacency matrix")
        return adj_matrix, nodes
    
    def cluster_ontology(self, n_clusters: int, random_state: int = 42, use_pruned: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Perform spectral clustering on the weighted ontology graph.
        
        Args:
            n_clusters: Number of clusters to create
            random_state: Random state for reproducibility
            use_pruned: Whether to use the pruned graph
            
        Returns:
            Tuple of (cluster labels, node list)
        """
        logger.info(f"Performing spectral clustering with {n_clusters} clusters")
        
        # Create weighted adjacency matrix
        adj_matrix, nodes = self.create_weighted_adjacency_matrix(use_pruned)
        
        # Perform spectral clustering
        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=random_state,
            n_init=10
        )
        
        cluster_labels = clusterer.fit_predict(adj_matrix)
        
        # Calculate silhouette score if possible
        if len(np.unique(cluster_labels)) > 1:
            sil_score = silhouette_score(adj_matrix, cluster_labels, metric='precomputed')
            logger.info(f"Silhouette score: {sil_score:.3f}")
        
        logger.info(f"Clustering complete. Created {len(np.unique(cluster_labels))} clusters")
        return cluster_labels, nodes
    
    def save_results(self, cluster_labels: np.ndarray, nodes: List[str], output_path: str):
        """Save clustering results to file.
        
        Args:
            cluster_labels: Cluster assignment for each node
            nodes: List of node IDs
            output_path: Path to save results
        """
        logger.info(f"Saving results to {output_path}")
        
        # Create results DataFrame
        results = []
        for node_id, cluster_id in zip(nodes, cluster_labels):
            node_data = self.graph.nodes[node_id]
            results.append({
                'node_id': node_id,
                'node_name': node_data.get('name', ''),
                'cluster_id': int(cluster_id),
                'abundance': self.node_weights.get(node_id, 1.0),
                'definition': node_data.get('definition', '')[:100] + '...' if len(node_data.get('definition', '')) > 200 else node_data.get('definition', '')
            })
        
        results_df = pd.DataFrame(results)
        
        # Sort by cluster and abundance
        results_df = results_df.sort_values(['cluster_id', 'abundance'], ascending=[True, False]).reset_index(drop=True)
        
        # Save to file
        if output_path.endswith('.csv'):
            results_df.to_csv(output_path, index=False)
        else:
            results_df.to_csv(output_path, sep='\t', index=False)
        
        # Print cluster summary
        cluster_summary = results_df.groupby('cluster_id').agg({
            'node_id': 'count',
            'abundance': ['sum', 'mean', 'std']
        }).round(3)
        
        print("\nCluster Summary:")
        print(cluster_summary)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Cluster MONDO disease ontology with abundance weighting",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "ontology_path",
        help="Path to MONDO OWL ontology file"
    )
    parser.add_argument(
        "abundance_path", 
        help="Path to abundance table (CSV/TSV with 'node_id' and 'node_abundance' columns)"
    )
    parser.add_argument(
        "-n", "--n-clusters",
        type=int,
        required=True,
        help="Number of clusters to create"
    )
    parser.add_argument(
        "-o", "--output",
        default="mondo_clusters.tsv",
        help="Output file path (default: mondo_clusters.tsv)"
    )
    parser.add_argument(
        "--prune-nodes",
        action="store_true",
        help="Remove nodes without abundance values and create direct edges between remaining nodes"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level (default: INFO)"
    )
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate inputs
    if args.n_clusters < 1:
        logger.error("Number of clusters must be >= 1")
        sys.exit(1)
        
    if not Path(args.ontology_path).exists():
        logger.error(f"Ontology file not found: {args.ontology_path}")
        sys.exit(1)
        
    if not Path(args.abundance_path).exists():
        logger.error(f"Abundance file not found: {args.abundance_path}")
        sys.exit(1)
    
    # Initialize clusterer
    clusterer = MondoOntologyClusterer(args.ontology_path, args.abundance_path)
    
    # Load data
    clusterer.load_ontology()
    clusterer.load_abundance_data()
    clusterer.create_node_weights()
    
    # Prune graph if requested
    if args.prune_nodes:
        clusterer.prune_graph()
    
    # Perform clustering
    cluster_labels, nodes = clusterer.cluster_ontology(
        args.n_clusters, 
        args.random_state,
        use_pruned=args.prune_nodes
    )
    
    # Save results
    clusterer.save_results(cluster_labels, nodes, args.output)
    
    print(f"\nClustering complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()