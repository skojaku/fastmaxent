# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FastMaxEnt Demo: Network Sampling with Karate Club Network
# 
# This notebook demonstrates how to use the **FastMaxEnt** package to generate random networks with specified expected degree and strength sequences. We'll use the classic Zachary's Karate Club network from NetworkX as our example.
# 
# ## Key Features Demonstrated:
# - **UBCM (Unweighted Binary Configuration Model)**: Generate unweighted random networks preserving degree sequences
# - **UECM (Undirected Enhanced Configuration Model)**: Generate weighted random networks preserving degree and strength sequences
# - **Network visualization**: Compare original vs sampled networks
# - **Statistical analysis**: Degree-degree and strength-strength correlations

# %% [markdown]
# ## 1. Setup and Imports

# %%
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# FastMaxEnt and NEMtropy imports
from fastmaxent import sampling
from NEMtropy import UndirectedGraph
from NEMtropy import network_functions

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
np.random.seed(42)  # For reproducibility

print("FastMaxEnt Demo - Network Sampling")
print("=" * 40)

# %% [markdown]
# ## 2. Load and Explore the Karate Club Network
# 
# The Zachary's Karate Club network is a classic example in network science. NetworkX provides this network with edge weights representing the strength of relationships.

# %%
# Load the Karate Club network (weighted version)
G = nx.karate_club_graph()

# Add weights to edges (NetworkX's karate club doesn't have weights by default)
# We'll create meaningful weights based on the network structure
np.random.seed(42)
for (u, v) in G.edges():
    # Weight based on sum of degrees with some noise
    base_weight = (G.degree[u] + G.degree[v]) / 10
    noise = np.random.normal(0, 0.5)
    weight = max(0.5, base_weight + noise)  # Ensure positive weights
    G[u][v]['weight'] = round(weight, 1)

# Convert to adjacency matrix for analysis
adj_matrix = nx.to_numpy_array(G, weight='weight')
n_nodes = len(G.nodes)
n_edges = len(G.edges)

print(f"Network Statistics:")
print(f"- Nodes: {n_nodes}")
print(f"- Edges: {n_edges}")
print(f"- Density: {nx.density(G):.3f}")
print(f"- Average degree: {np.mean([d for n, d in G.degree()]):.2f}")
print(f"- Weight range: {min(nx.get_edge_attributes(G, 'weight').values()):.1f} - {max(nx.get_edge_attributes(G, 'weight').values()):.1f}")

# %% [markdown]
# ### Visualize the Original Network

# %%
# Create a comprehensive visualization of the original network
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Network visualization
pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]

# Plot 1: Network structure with edge weights
nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                      node_size=300, alpha=0.7, ax=axes[0])
nx.draw_networkx_edges(G, pos, width=[w/2 for w in weights], 
                      alpha=0.6, edge_color='gray', ax=axes[0])
nx.draw_networkx_labels(G, pos, font_size=8, ax=axes[0])
axes[0].set_title('Karate Club Network\n(Edge thickness = weight)')
axes[0].axis('off')

# Plot 2: Degree distribution
degrees = [d for n, d in G.degree()]
axes[1].hist(degrees, bins=range(min(degrees), max(degrees)+2), 
             alpha=0.7, color='skyblue', edgecolor='black')
axes[1].set_xlabel('Degree')
axes[1].set_ylabel('Count')
axes[1].set_title('Degree Distribution')
axes[1].grid(True, alpha=0.3)

# Plot 3: Strength distribution (weighted degrees)
strengths = [sum(G[n][neighbor]['weight'] for neighbor in G[n]) for n in G.nodes()]
axes[2].hist(strengths, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
axes[2].set_xlabel('Strength (weighted degree)')
axes[2].set_ylabel('Count')
axes[2].set_title('Strength Distribution')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Store original properties for comparison
original_degrees = np.array(degrees)
original_strengths = np.array(strengths)

print(f"\nOriginal Network Properties:")
print(f"Degree sequence: {original_degrees}")
print(f"Strength sequence: {np.round(original_strengths, 1)}")

# %% [markdown]
# ## 3. UBCM Demo: Unweighted Network Sampling
# 
# First, let's demonstrate sampling unweighted networks using the **Unweighted Binary Configuration Model (UBCM)**. This preserves the expected degree sequence while ignoring edge weights.

# %%
print("UBCM Demo: Unweighted Network Sampling")
print("=" * 40)

# Convert weighted network to unweighted (binary) adjacency matrix
unweighted_adj = (adj_matrix > 0).astype(int)

# Fit UBCM parameters using NEMtropy
print("Fitting UBCM parameters...")
ubcm_graph = UndirectedGraph(unweighted_adj)
ubcm_graph.solve_tool(model="cm", method="quasinewton", initial_guess="random", tol=1e-08)

# Extract alpha parameters for FastMaxEnt
alphas = -np.log(ubcm_graph.x)
print(f"Fitted alpha parameters: {np.round(alphas, 3)}")

# Generate multiple samples using FastMaxEnt
print("Generating unweighted network samples...")
n_samples = 5
ubcm_samples = sampling(alpha=alphas, weighted=False, n_samples=n_samples)

print(f"\nGenerated {len(ubcm_samples)} unweighted network samples")

# Analyze the samples
sample_degrees = []
for i, edges in enumerate(ubcm_samples):
    if len(edges) > 0:
        edges_array = np.array(edges)
        # Create adjacency matrix from edge list
        sample_adj = np.zeros((n_nodes, n_nodes))
        for edge in edges_array:
            sample_adj[edge[0], edge[1]] = 1
            sample_adj[edge[1], edge[0]] = 1
        
        degrees = np.sum(sample_adj, axis=0).astype(int)
        sample_degrees.append(degrees)
        print(f"Sample {i+1}: {len(edges)} edges, degree sequence: {degrees}")
    else:
        sample_degrees.append(np.zeros(n_nodes))
        print(f"Sample {i+1}: 0 edges (empty network)")

# %% [markdown]
# ### Visualize UBCM Results

# %%
# Create visualization comparing original and sampled networks
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot original unweighted network
G_unweighted = nx.from_numpy_array(unweighted_adj)
nx.draw_networkx_nodes(G_unweighted, pos, node_color='lightblue', 
                      node_size=300, alpha=0.7, ax=axes[0,0])
nx.draw_networkx_edges(G_unweighted, pos, alpha=0.6, edge_color='gray', ax=axes[0,0])
nx.draw_networkx_labels(G_unweighted, pos, font_size=8, ax=axes[0,0])
axes[0,0].set_title('Original (Unweighted)')
axes[0,0].axis('off')

# Plot first two samples
for i in range(min(2, len(ubcm_samples))):
    if len(ubcm_samples[i]) > 0:
        edges_array = np.array(ubcm_samples[i])
        G_sample = nx.Graph()
        G_sample.add_nodes_from(range(n_nodes))
        G_sample.add_edges_from(edges_array)
        
        nx.draw_networkx_nodes(G_sample, pos, node_color='lightgreen', 
                              node_size=300, alpha=0.7, ax=axes[0,i+1])
        nx.draw_networkx_edges(G_sample, pos, alpha=0.6, edge_color='gray', ax=axes[0,i+1])
        nx.draw_networkx_labels(G_sample, pos, font_size=8, ax=axes[0,i+1])
    axes[0,i+1].set_title(f'UBCM Sample {i+1}')
    axes[0,i+1].axis('off')

# Degree comparison plots
# Average degrees across samples
if sample_degrees:
    avg_sample_degrees = np.mean(sample_degrees, axis=0)
    
    # Scatter plot: original vs expected degrees
    axes[1,0].scatter(original_degrees, avg_sample_degrees, alpha=0.7, s=50)
    axes[1,0].plot([0, max(original_degrees)], [0, max(original_degrees)], 'r--', alpha=0.7)
    axes[1,0].set_xlabel('Original Degree')
    axes[1,0].set_ylabel('Average Sample Degree')
    axes[1,0].set_title('Degree Preservation (UBCM)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Degree distribution comparison
    axes[1,1].hist(original_degrees, bins=range(min(original_degrees), max(original_degrees)+2),
                   alpha=0.7, label='Original', color='blue', density=True)
    axes[1,1].hist(avg_sample_degrees, bins=range(int(min(avg_sample_degrees)), int(max(avg_sample_degrees))+2),
                   alpha=0.7, label='Samples (avg)', color='red', density=True)
    axes[1,1].set_xlabel('Degree')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Degree Distribution Comparison')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Correlation coefficient
    corr_coef = np.corrcoef(original_degrees, avg_sample_degrees)[0,1]
    axes[1,2].text(0.1, 0.9, f'Degree Correlation: {corr_coef:.4f}', 
                   transform=axes[1,2].transAxes, fontsize=12, fontweight='bold')
    axes[1,2].text(0.1, 0.8, f'Original avg degree: {np.mean(original_degrees):.2f}', 
                   transform=axes[1,2].transAxes, fontsize=10)
    axes[1,2].text(0.1, 0.7, f'Sample avg degree: {np.mean(avg_sample_degrees):.2f}', 
                   transform=axes[1,2].transAxes, fontsize=10)
    axes[1,2].text(0.1, 0.6, f'Max abs difference: {np.max(np.abs(original_degrees - avg_sample_degrees)):.3f}', 
                   transform=axes[1,2].transAxes, fontsize=10)
    axes[1,2].axis('off')
    axes[1,2].set_title('UBCM Statistics')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. UECM Demo: Weighted Network Sampling
# 
# Now let's demonstrate the **Undirected Enhanced Configuration Model (UECM)** for weighted networks. This preserves both the expected degree and strength sequences.

# %%
print("UECM Demo: Weighted Network Sampling")
print("=" * 40)

# Fit UECM parameters using NEMtropy
print("Fitting UECM parameters...")
uecm_graph = UndirectedGraph(adj_matrix)
uecm_graph.solve_tool(model="ecm", method="quasinewton", initial_guess="random", tol=1e-08)

# Extract alpha and beta parameters for FastMaxEnt
alphas_weighted = -np.log(uecm_graph.x)
betas_weighted = -np.log(uecm_graph.y)

print(f"Fitted alpha parameters: {np.round(alphas_weighted, 3)}")
print(f"Fitted beta parameters: {np.round(betas_weighted, 3)}")

# Generate weighted network samples using FastMaxEnt
print("Generating weighted network samples...")
n_samples_weighted = 5
uecm_samples = sampling(alpha=alphas_weighted, beta=betas_weighted, 
                       weighted=True, n_samples=n_samples_weighted)

print(f"\nGenerated {len(uecm_samples)} weighted network samples")

# Analyze the weighted samples
sample_degrees_weighted = []
sample_strengths_weighted = []

for i, edges in enumerate(uecm_samples):
    if len(edges) > 0:
        edges_array = np.array(edges)
        
        # Create weighted adjacency matrix from edge list
        sample_adj_weighted = np.zeros((n_nodes, n_nodes))
        for edge in edges_array:
            sample_adj_weighted[edge[0], edge[1]] = edge[2]
            sample_adj_weighted[edge[1], edge[0]] = edge[2]
        
        degrees = np.sum(sample_adj_weighted > 0, axis=0).astype(int)
        strengths = np.sum(sample_adj_weighted, axis=0)
        
        sample_degrees_weighted.append(degrees)
        sample_strengths_weighted.append(strengths)
        
        print(f"Sample {i+1}: {len(edges)} edges")
        print(f"  Degrees: {degrees}")
        print(f"  Strengths: {np.round(strengths, 1)}")
    else:
        sample_degrees_weighted.append(np.zeros(n_nodes))
        sample_strengths_weighted.append(np.zeros(n_nodes))
        print(f"Sample {i+1}: 0 edges (empty network)")

# %% [markdown]
# ### Visualize UECM Results

# %%
# Create comprehensive visualization for weighted networks
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# Row 1: Network visualizations
# Original weighted network
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                      node_size=300, alpha=0.7, ax=axes[0,0])
nx.draw_networkx_edges(G, pos, width=[w/2 for w in weights], 
                      alpha=0.6, edge_color='gray', ax=axes[0,0])
nx.draw_networkx_labels(G, pos, font_size=8, ax=axes[0,0])
axes[0,0].set_title('Original Weighted Network')
axes[0,0].axis('off')

# First two weighted samples
for i in range(min(2, len(uecm_samples))):
    if len(uecm_samples[i]) > 0:
        edges_array = np.array(uecm_samples[i])
        G_sample = nx.Graph()
        G_sample.add_nodes_from(range(n_nodes))
        for edge in edges_array:
            G_sample.add_edge(edge[0], edge[1], weight=edge[2])
        
        sample_weights = [G_sample[u][v]['weight'] for u, v in G_sample.edges()]
        nx.draw_networkx_nodes(G_sample, pos, node_color='lightcoral', 
                              node_size=300, alpha=0.7, ax=axes[0,i+1])
        nx.draw_networkx_edges(G_sample, pos, width=[w/3 for w in sample_weights], 
                              alpha=0.6, edge_color='gray', ax=axes[0,i+1])
        nx.draw_networkx_labels(G_sample, pos, font_size=8, ax=axes[0,i+1])
    axes[0,i+1].set_title(f'UECM Sample {i+1}')
    axes[0,i+1].axis('off')

# Row 2: Degree analysis
if sample_degrees_weighted:
    avg_sample_degrees_weighted = np.mean(sample_degrees_weighted, axis=0)
    
    # Degree scatter plot
    axes[1,0].scatter(original_degrees, avg_sample_degrees_weighted, alpha=0.7, s=50, color='blue')
    axes[1,0].plot([0, max(original_degrees)], [0, max(original_degrees)], 'r--', alpha=0.7)
    axes[1,0].set_xlabel('Original Degree')
    axes[1,0].set_ylabel('Average Sample Degree')
    axes[1,0].set_title('Degree Preservation (UECM)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Degree distribution comparison
    axes[1,1].hist(original_degrees, bins=range(min(original_degrees), max(original_degrees)+2),
                   alpha=0.7, label='Original', color='blue', density=True)
    axes[1,1].hist(avg_sample_degrees_weighted, bins=range(int(min(avg_sample_degrees_weighted)), int(max(avg_sample_degrees_weighted))+2),
                   alpha=0.7, label='Samples (avg)', color='red', density=True)
    axes[1,1].set_xlabel('Degree')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Degree Distribution (UECM)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Degree statistics
    deg_corr = np.corrcoef(original_degrees, avg_sample_degrees_weighted)[0,1]
    axes[1,2].text(0.1, 0.9, f'Degree Correlation: {deg_corr:.4f}', 
                   transform=axes[1,2].transAxes, fontsize=11, fontweight='bold')
    axes[1,2].text(0.1, 0.8, f'Original avg: {np.mean(original_degrees):.2f}', 
                   transform=axes[1,2].transAxes, fontsize=9)
    axes[1,2].text(0.1, 0.7, f'Sample avg: {np.mean(avg_sample_degrees_weighted):.2f}', 
                   transform=axes[1,2].transAxes, fontsize=9)
    axes[1,2].text(0.1, 0.6, f'Max abs diff: {np.max(np.abs(original_degrees - avg_sample_degrees_weighted)):.3f}', 
                   transform=axes[1,2].transAxes, fontsize=9)
    axes[1,2].axis('off')
    axes[1,2].set_title('Degree Statistics')

# Row 3: Strength analysis
if sample_strengths_weighted:
    avg_sample_strengths_weighted = np.mean(sample_strengths_weighted, axis=0)
    
    # Strength scatter plot
    axes[2,0].scatter(original_strengths, avg_sample_strengths_weighted, alpha=0.7, s=50, color='green')
    axes[2,0].plot([0, max(original_strengths)], [0, max(original_strengths)], 'r--', alpha=0.7)
    axes[2,0].set_xlabel('Original Strength')
    axes[2,0].set_ylabel('Average Sample Strength')
    axes[2,0].set_title('Strength Preservation (UECM)')
    axes[2,0].grid(True, alpha=0.3)
    
    # Strength distribution comparison
    axes[2,1].hist(original_strengths, bins=15, alpha=0.7, label='Original', color='green', density=True)
    axes[2,1].hist(avg_sample_strengths_weighted, bins=15, alpha=0.7, label='Samples (avg)', color='orange', density=True)
    axes[2,1].set_xlabel('Strength')
    axes[2,1].set_ylabel('Density')
    axes[2,1].set_title('Strength Distribution (UECM)')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    
    # Strength statistics
    str_corr = np.corrcoef(original_strengths, avg_sample_strengths_weighted)[0,1]
    axes[2,2].text(0.1, 0.9, f'Strength Correlation: {str_corr:.4f}', 
                   transform=axes[2,2].transAxes, fontsize=11, fontweight='bold')
    axes[2,2].text(0.1, 0.8, f'Original avg: {np.mean(original_strengths):.2f}', 
                   transform=axes[2,2].transAxes, fontsize=9)
    axes[2,2].text(0.1, 0.7, f'Sample avg: {np.mean(avg_sample_strengths_weighted):.2f}', 
                   transform=axes[2,2].transAxes, fontsize=9)
    axes[2,2].text(0.1, 0.6, f'Max abs diff: {np.max(np.abs(original_strengths - avg_sample_strengths_weighted)):.3f}', 
                   transform=axes[2,2].transAxes, fontsize=9)
    axes[2,2].axis('off')
    axes[2,2].set_title('Strength Statistics')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Statistical Analysis and Convergence
# 
# Let's analyze how well our sampling preserves the network constraints and examine the statistical properties across multiple samples.

# %%
print("Statistical Analysis")
print("=" * 40)

# Generate more samples for better statistics
print("Generating additional samples for statistical analysis...")
n_analysis_samples = 20

# UBCM additional samples
ubcm_analysis_samples = sampling(alpha=alphas, weighted=False, n_samples=n_analysis_samples)

# UECM additional samples  
uecm_analysis_samples = sampling(alpha=alphas_weighted, beta=betas_weighted, 
                                weighted=True, n_samples=n_analysis_samples)

# %% [markdown]
# ### Constraint Preservation Analysis

# %%
# Analyze constraint preservation across many samples
ubcm_all_degrees = []
uecm_all_degrees = []
uecm_all_strengths = []

# Process UBCM samples
for edges in ubcm_analysis_samples:
    if len(edges) > 0:
        edges_array = np.array(edges)
        sample_adj = np.zeros((n_nodes, n_nodes))
        for edge in edges_array:
            sample_adj[edge[0], edge[1]] = 1
            sample_adj[edge[1], edge[0]] = 1
        degrees = np.sum(sample_adj, axis=0).astype(int)
        ubcm_all_degrees.append(degrees)
    else:
        ubcm_all_degrees.append(np.zeros(n_nodes, dtype=int))

# Process UECM samples
for edges in uecm_analysis_samples:
    if len(edges) > 0:
        edges_array = np.array(edges)
        sample_adj = np.zeros((n_nodes, n_nodes))
        for edge in edges_array:
            sample_adj[edge[0], edge[1]] = edge[2]
            sample_adj[edge[1], edge[0]] = edge[2]
        degrees = np.sum(sample_adj > 0, axis=0).astype(int)
        strengths = np.sum(sample_adj, axis=0)
        uecm_all_degrees.append(degrees)
        uecm_all_strengths.append(strengths)
    else:
        uecm_all_degrees.append(np.zeros(n_nodes, dtype=int))
        uecm_all_strengths.append(np.zeros(n_nodes))

# Convert to arrays for analysis
ubcm_all_degrees = np.array(ubcm_all_degrees)
uecm_all_degrees = np.array(uecm_all_degrees) 
uecm_all_strengths = np.array(uecm_all_strengths)

# Calculate statistics
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# UBCM degree convergence
ubcm_mean_degrees = np.mean(ubcm_all_degrees, axis=0)
ubcm_std_degrees = np.std(ubcm_all_degrees, axis=0)

axes[0,0].errorbar(range(n_nodes), ubcm_mean_degrees, yerr=ubcm_std_degrees, 
                   fmt='o', alpha=0.7, label='UBCM samples', color='blue')
axes[0,0].scatter(range(n_nodes), original_degrees, color='red', s=50, 
                  label='Original', marker='s', alpha=0.8)
axes[0,0].set_xlabel('Node ID')
axes[0,0].set_ylabel('Degree')
axes[0,0].set_title('UBCM: Degree Preservation')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# UECM degree convergence
uecm_mean_degrees = np.mean(uecm_all_degrees, axis=0)
uecm_std_degrees = np.std(uecm_all_degrees, axis=0)

axes[0,1].errorbar(range(n_nodes), uecm_mean_degrees, yerr=uecm_std_degrees, 
                   fmt='o', alpha=0.7, label='UECM samples', color='green')
axes[0,1].scatter(range(n_nodes), original_degrees, color='red', s=50, 
                  label='Original', marker='s', alpha=0.8)
axes[0,1].set_xlabel('Node ID')
axes[0,1].set_ylabel('Degree')
axes[0,1].set_title('UECM: Degree Preservation')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# UECM strength convergence
uecm_mean_strengths = np.mean(uecm_all_strengths, axis=0)
uecm_std_strengths = np.std(uecm_all_strengths, axis=0)

axes[0,2].errorbar(range(n_nodes), uecm_mean_strengths, yerr=uecm_std_strengths, 
                   fmt='o', alpha=0.7, label='UECM samples', color='purple')
axes[0,2].scatter(range(n_nodes), original_strengths, color='red', s=50, 
                  label='Original', marker='s', alpha=0.8)
axes[0,2].set_xlabel('Node ID')
axes[0,2].set_ylabel('Strength')
axes[0,2].set_title('UECM: Strength Preservation')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# Distribution of correlations across samples
ubcm_correlations = [np.corrcoef(original_degrees, deg_seq)[0,1] 
                     for deg_seq in ubcm_all_degrees if not np.isnan(np.corrcoef(original_degrees, deg_seq)[0,1])]
uecm_deg_correlations = [np.corrcoef(original_degrees, deg_seq)[0,1] 
                         for deg_seq in uecm_all_degrees if not np.isnan(np.corrcoef(original_degrees, deg_seq)[0,1])]
uecm_str_correlations = [np.corrcoef(original_strengths, str_seq)[0,1] 
                         for str_seq in uecm_all_strengths if not np.isnan(np.corrcoef(original_strengths, str_seq)[0,1])]

# Correlation distributions
axes[1,0].hist(ubcm_correlations, bins=15, alpha=0.7, color='blue', edgecolor='black')
axes[1,0].set_xlabel('Degree Correlation')
axes[1,0].set_ylabel('Count')
axes[1,0].set_title(f'UBCM Degree Correlations\n(mean: {np.mean(ubcm_correlations):.3f})')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].hist(uecm_deg_correlations, bins=15, alpha=0.7, color='green', edgecolor='black')
axes[1,1].set_xlabel('Degree Correlation')
axes[1,1].set_ylabel('Count')
axes[1,1].set_title(f'UECM Degree Correlations\n(mean: {np.mean(uecm_deg_correlations):.3f})')
axes[1,1].grid(True, alpha=0.3)

axes[1,2].hist(uecm_str_correlations, bins=15, alpha=0.7, color='purple', edgecolor='black')
axes[1,2].set_xlabel('Strength Correlation')
axes[1,2].set_ylabel('Count')
axes[1,2].set_title(f'UECM Strength Correlations\n(mean: {np.mean(uecm_str_correlations):.3f})')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"\nConstraint Preservation Summary:")
print(f"UBCM Degree Correlation: {np.mean(ubcm_correlations):.4f} ± {np.std(ubcm_correlations):.4f}")
print(f"UECM Degree Correlation: {np.mean(uecm_deg_correlations):.4f} ± {np.std(uecm_deg_correlations):.4f}")
print(f"UECM Strength Correlation: {np.mean(uecm_str_correlations):.4f} ± {np.std(uecm_str_correlations):.4f}")

# %% [markdown]
# ## 6. Summary and Conclusions
# 
# This demonstration shows how **FastMaxEnt** can efficiently generate random networks that preserve important structural properties:

# %%
print("FastMaxEnt Demo Summary")
print("=" * 50)
print(f"Original Karate Club Network:")
print(f"  - Nodes: {n_nodes}")
print(f"  - Edges: {n_edges}")
print(f"  - Average degree: {np.mean(original_degrees):.2f}")
print(f"  - Average strength: {np.mean(original_strengths):.2f}")

print(f"\nUBCM (Unweighted) Results:")
print(f"  - Generated {len(ubcm_analysis_samples)} samples")
print(f"  - Average degree correlation: {np.mean(ubcm_correlations):.4f}")
print(f"  - Degree preservation: Excellent")

print(f"\nUECM (Weighted) Results:")
print(f"  - Generated {len(uecm_analysis_samples)} samples")
print(f"  - Average degree correlation: {np.mean(uecm_deg_correlations):.4f}")
print(f"  - Average strength correlation: {np.mean(uecm_str_correlations):.4f}")
print(f"  - Both degree and strength preservation: Excellent")

print(f"\nKey FastMaxEnt Advantages:")
print(f"  ✓ Exact sampling from maximum entropy distribution")
print(f"  ✓ No MCMC bias or convergence issues")
print(f"  ✓ Fast geometric jump sampling algorithm")
print(f"  ✓ Seamless integration with NEMtropy for parameter fitting")
print(f"  ✓ Support for both weighted and unweighted networks")

# %% [markdown]
# ### Key Takeaways:
# 
# 1. **UBCM** preserves degree sequences in unweighted networks
# 2. **UECM** preserves both degree and strength sequences in weighted networks
# 3. **FastMaxEnt** provides unbiased sampling from the maximum entropy distribution
# 4. The **geometric jump** algorithm makes sampling very efficient
# 5. Integration with **NEMtropy** makes it easy to fit parameters from real data
# 
# This makes FastMaxEnt ideal for:
# - **Network null models** for statistical testing
# - **Randomized network analysis** preserving key properties
# - **Studying network structure** through controlled randomization
# - **Benchmarking** against maximum entropy ensembles

print("\nDemo completed successfully! 🎉")