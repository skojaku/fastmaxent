#!/usr/bin/env python3
"""
FastMaxEnt CLI Tool

A command-line interface for inferring network parameters from edge tables
and generating random network samples using FastMaxEnt.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

try:
    from NEMtropy import UndirectedGraph
    from NEMtropy import network_functions
    NEMTROPY_AVAILABLE = True
except ImportError:
    NEMTROPY_AVAILABLE = False

from fastmaxent import sampling


def load_edge_table(filepath, delimiter=',', weighted=False):
    """
    Load edge table from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the edge table
    delimiter : str
        CSV delimiter character
    weighted : bool
        Whether the network is weighted
        
    Returns
    -------
    pandas.DataFrame
        Edge table with columns [source, target] or [source, target, weight]
    """
    try:
        df = pd.read_csv(filepath, delimiter=delimiter)
    except Exception as e:
        raise ValueError(f"Error reading CSV file {filepath}: {e}")
    
    # Validate columns
    if weighted:
        if len(df.columns) < 3:
            raise ValueError("Weighted network requires at least 3 columns: source, target, weight")
        df.columns = ['source', 'target', 'weight']
    else:
        if len(df.columns) < 2:
            raise ValueError("Network requires at least 2 columns: source, target")
        df.columns = ['source', 'target'] + [f'col_{i}' for i in range(len(df.columns) - 2)]
        df = df[['source', 'target']]
    
    return df


def infer_parameters(edge_df, weighted=False):
    """
    Infer UBCM or UECM parameters from edge table using NEMtropy.
    
    Parameters
    ----------
    edge_df : pandas.DataFrame
        Edge table
    weighted : bool
        Whether to fit weighted (UECM) or unweighted (UBCM) model
        
    Returns
    -------
    tuple
        (alpha, beta) parameters. beta is None for unweighted networks.
    """
    if not NEMTROPY_AVAILABLE:
        raise ImportError(
            "NEMtropy is required for parameter inference. "
            "Install with: pip install nemtropy>=3.0.0"
        )
    
    # Convert to numpy array for NEMtropy
    edge_array = edge_df.values
    
    # Build adjacency matrix
    adjacency_matrix = network_functions.build_adjacency_from_edgelist(
        edge_array, 
        is_directed=False, 
        is_weighted=weighted, 
        is_sparse=False
    )
    
    # Create graph and solve
    graph = UndirectedGraph(adjacency_matrix)
    
    if weighted:
        print("Fitting UECM (weighted) model...")
        model = "ecm"
    else:
        print("Fitting UBCM (unweighted) model...")
        model = "cm"
    
    graph.solve_tool(
        model=model, 
        method="quasinewton", 
        initial_guess="random", 
        tol=1e-08
    )
    
    # Extract parameters
    alpha = -np.log(graph.x)
    beta = -np.log(graph.y) if weighted else None
    
    print(f"Successfully fitted {model.upper()} model")
    print(f"Number of nodes: {len(alpha)}")
    
    return alpha, beta


def generate_networks(alpha, beta, n_samples, weighted=False):
    """
    Generate random network samples using FastMaxEnt.
    
    Parameters
    ----------
    alpha : numpy.ndarray
        Alpha parameters
    beta : numpy.ndarray or None
        Beta parameters (None for unweighted)
    n_samples : int
        Number of samples to generate
    weighted : bool
        Whether to generate weighted networks
        
    Returns
    -------
    list
        List of network edge lists
    """
    print(f"Generating {n_samples} network samples...")
    
    networks = sampling(
        alpha=alpha,
        beta=beta,
        weighted=weighted,
        n_samples=n_samples
    )
    
    print(f"Successfully generated {len(networks)} networks")
    return networks


def save_networks(networks, output_dir, weighted=False):
    """
    Save network samples to CSV files.
    
    Parameters
    ----------
    networks : list
        List of network edge lists
    output_dir : str
        Output directory path
    weighted : bool
        Whether networks are weighted
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, network in enumerate(networks):
        if len(network) == 0:
            print(f"Warning: Network sample {i+1} has no edges")
            continue
            
        # Convert to DataFrame
        network_array = np.array(network)
        
        if weighted:
            df = pd.DataFrame(network_array, columns=['source', 'target', 'weight'])
        else:
            df = pd.DataFrame(network_array, columns=['source', 'target'])
        
        # Save to CSV
        output_path = os.path.join(output_dir, f'network_sample_{i+1:03d}.csv')
        df.to_csv(output_path, index=False)
        
        print(f"Saved network sample {i+1} to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Infer parameters from edge table and generate random networks using FastMaxEnt"
    )
    
    parser.add_argument(
        'input',
        help='Path to input CSV edge table file'
    )
    
    parser.add_argument(
        'output_dir',
        help='Output directory for generated network samples'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of random networks to generate (default: 10)'
    )
    
    parser.add_argument(
        '--weighted',
        action='store_true',
        help='Treat network as weighted (use UECM model)'
    )
    
    parser.add_argument(
        '--delimiter',
        default=',',
        help='CSV delimiter character (default: comma)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    try:
        # Load edge table
        print(f"Loading edge table from {args.input}...")
        edge_df = load_edge_table(args.input, args.delimiter, args.weighted)
        print(f"Loaded {len(edge_df)} edges")
        
        # Infer parameters
        alpha, beta = infer_parameters(edge_df, args.weighted)
        
        # Generate networks
        networks = generate_networks(alpha, beta, args.n_samples, args.weighted)
        
        # Save networks
        save_networks(networks, args.output_dir, args.weighted)
        
        print(f"\nCompleted! Generated {len(networks)} networks in {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()