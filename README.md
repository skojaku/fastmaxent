# Fast Unbiased Sampling of Networks with Given Expected Degrees and Strengths
This repo features code for Fast unbiased sampling of networks with given expected degrees and strengths paper.

## Installation

To use the `fastmaxent` package, you'll need Python 3.9 or higher.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EKUL-Skywalker/fast_unbiased_sampling_of_networks_with_given_expected_degrees_and_strengths.git
    cd fast_unbiased_sampling_of_networks_with_given_expected_degrees_and_strengths
    ```
2.  **Install the package:**
    This will install the `fastmaxent` module and all its dependencies (`numpy`, `numba`, `nemtropy`).
    ```bash
    pip install .
    ```

## Citation
If you find this code useful for your research, please cite our paper:

```bibtex
@article{Li2025Fast,
  title={Fast unbiased sampling of networks with given expected degrees and strengths},
  author={Xuanchi Li and Xin Wang and Sadamori Kojaku},
  year={2025}
}
```
## Dependencies
  - python=3.9
  - snakemake=7.32.4
  - numpy=1.24.4
  - numba=0.57.1
  - nemtropy=3.0.3