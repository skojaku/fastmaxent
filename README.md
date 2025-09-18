# FastMaxEnt: Fast Unbiased Sampling of Networks with Given Expected Degrees and Strengths

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FastMaxEnt provides efficient algorithms for sampling networks from maximum entropy models with given expected degree and strength sequences. This package implements fast rejection sampling algorithms for both unweighted (UBCM) and weighted (UECM) configuration models.

## What is FastMaxEnt?

FastMaxEnt implements fast and unbiased algorithms for generating network samples that:
- Preserve expected network properties
- Are statistically unbiased
- Scale efficiently
- Handle both weighted and unweighted networks

## Installation

### Using pip

#### From GitHub (Recommended)
```bash
pip install git+https://github.com/EKUL-Skywalker/fast_unbiased_sampling_of_networks_with_given_expected_degrees_and_strengths.git
```

#### With Inference Dependencies
If you need parameter fitting capabilities (NEMtropy integration):
```bash
pip install git+https://github.com/EKUL-Skywalker/fast_unbiased_sampling_of_networks_with_given_expected_degrees_and_strengths.git[inference]
```

### Using uv (Faster Alternative)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

#### From GitHub
```bash
uv add git+https://github.com/EKUL-Skywalker/fast_unbiased_sampling_of_networks_with_given_expected_degrees_and_strengths.git
```

#### With Inference Dependencies
```bash
uv add git+https://github.com/EKUL-Skywalker/fast_unbiased_sampling_of_networks_with_given_expected_degrees_and_strengths.git[inference]
```

## Quick Start

### Unweighted Networks (UBCM)
```python
import numpy as np
from fastmaxent import sampling

# Define fitness parameters (alpha = -log(theta))
alpha = np.array([1.5, 2.0, 1.0, 2.5])

# Sample unweighted networks
networks = sampling(alpha, weighted=False, n_samples=3)

# Each network is a list of edges [source, target]
print(f"First network: {networks[0]}")
```

### Weighted Networks (UECM)
```python
import numpy as np
from fastmaxent import sampling

# Define degree and strength parameters
alpha = np.array([1.5, 2.0, 1.0, 2.5])  # degree constraints
beta = np.array([0.5, 1.0, 0.8, 1.2])   # strength constraints

# Sample weighted networks
networks = sampling(alpha, beta=beta, weighted=True, n_samples=2)

# Each network is a list of edges [source, target, weight]
print(f"First weighted network: {networks[0]}")
```

## Usage with NEMtropy

FastMaxEnt is designed to work seamlessly with [NEMtropy](https://github.com/nicoloval/NEMtropy) for parameter fitting:

### Unweighted Example
```python
import numpy as np
import pandas as pd
from NEMtropy import UndirectedGraph
from NEMtropy import network_functions
from fastmaxent import sampling

# Load your network data
df = pd.read_csv("your_network.csv").values
adjacency_matrix = network_functions.build_adjacency_from_edgelist(
    df, is_directed=False, is_weighted=False, is_sparse=False
)

# Fit UBCM parameters using NEMtropy
graph = UndirectedGraph(adjacency_matrix)
graph.solve_tool(model="cm", method="quasinewton", initial_guess="random", tol=1e-08)

# Extract parameters for FastMaxEnt
alphas = -np.log(graph.x)

# Sample networks preserving degree sequence
networks = sampling(alphas, weighted=False, n_samples=100)
```

### Weighted Example
```python
import numpy as np
import pandas as pd
from NEMtropy import UndirectedGraph
from NEMtropy import network_functions
from fastmaxent import sampling

# Load weighted network data
df = pd.read_csv("your_weighted_network.csv").values
adjacency_matrix = network_functions.build_adjacency_from_edgelist(
    df, is_directed=False, is_weighted=True, is_sparse=False
)

# Fit UECM parameters using NEMtropy
graph = UndirectedGraph(adjacency_matrix)
graph.solve_tool(model="ecm", method="quasinewton", initial_guess="random", tol=1e-08)

# Extract parameters for FastMaxEnt
alphas = -np.log(graph.x)
betas = -np.log(graph.y)

# Sample networks preserving degree and strength sequences
networks = sampling(alphas, beta=betas, weighted=True, n_samples=100)
```

## API Reference

### `sampling(alpha, beta=None, weighted=False, n_samples=1)`

Generate network samples from UBCM or UECM models.

**Parameters:**
- `alpha` (numpy.ndarray): Degree constraint parameters for each node
- `beta` (numpy.ndarray, optional): Strength constraint parameters (required for weighted=True)
- `weighted` (bool): Whether to sample weighted networks (default: False)
- `n_samples` (int): Number of network samples to generate (default: 1)

**Returns:**
- `list`: List of network edge lists. For unweighted: `[source, target]`. For weighted: `[source, target, weight]`

## Examples

Complete working examples are provided in the `examples/` directory:

- **[`examples/example.ipynb`](examples/example.ipynb)** - Comprehensive Jupyter notebook tutorial using Zachary's Karate Club network
  - Demonstrates both UBCM (unweighted) and UECM (weighted) sampling
  - Shows parameter fitting with NEMtropy
  - Includes degree and strength preservation verification
  - Visualization of results

- [`examples/demo_ubcm.py`](examples/demo_ubcm.py) - Unweighted network sampling script
- [`examples/demo_uecm.py`](examples/demo_uecm.py) - Weighted network sampling script

### Running Examples

#### Python Scripts:
```bash
cd examples
python demo_ubcm.py   # Unweighted network demo
python demo_uecm.py   # Weighted network demo
```

#### Jupyter Notebook:
```bash
cd examples
jupyter notebook example.ipynb   # Interactive tutorial
```

## Citation

If you use FastMaxEnt in your research, please cite our paper:

```bibtex
@article{li2025fast,
  title={Fast unbiased sampling of networks with given expected degrees and strengths},
  author={Xuanchi Li and Xin Wang and Sadamori Kojaku},
  journal={arXiv:2509.13230},
  year={2025}
}
```

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.20.0
- numba ≥ 0.56.0
- Optional: nemtropy ≥ 3.0.0 (for parameter fitting examples)