"""
FastMaxEnt Network Sampling Module

This module implements fast and unbiased sampling algorithms for network generation
based on given expected degrees and strengths. It provides implementations for:

1. Undirected Binary Configuration Model (UBCM) - for unweighted networks
2. Undirected Enhanced Configuration Model (UECM) - for weighted networks

The algorithms use rejection sampling with geometric jumps to achieve efficient
sampling while maintaining statistical accuracy.

References:
    Li, X., Wang, X., & Kojaku, S. (2025). Fast unbiased sampling of networks
    with given expected degrees and strengths. arXiv:2509.13230.
"""

import numpy as np
from numba import njit, jit


def sampling(
    alpha,
    beta=None,
    weighted=False,
    n_samples=1,
    flatten=False,
    *params,
    **kwargs,
):
    """
    Sample networks from UBCM or UECM models with given parameters.

    This function generates network samples using fast rejection sampling algorithms
    for either unweighted (UBCM) or weighted (UECM) configuration models.

    Parameters
    ----------
    alpha : numpy.ndarray
        Array of alpha parameters for each node. These parameters control the
        expected degree of each node in the sampled networks.
        - For UBCM: alpha[i] = -log(theta[i]) where theta[i] is the fitness parameter
        - For UECM: alpha[i] controls degree constraints alongside beta[i]

    beta : numpy.ndarray, optional
        Array of beta parameters for each node (required for weighted=True).
        These parameters control the expected strength of each node in weighted networks.
        Must have the same length as alpha when provided.

    weighted : bool, default=False
        Whether to sample weighted networks using UECM (True) or unweighted networks
        using UBCM (False).

    n_samples : int, default=1
        Number of network samples to generate. Must be greater than 0.

    flatten : bool, default=False
        Whether to flatten multiple samples into a single edge list.
        - If True: returns single flattened list with all edges containing network ID in last column
        - If False: returns list of n_samples separate edge lists (original behavior)

    Returns
    -------
    list
        If flatten=False (default): List containing n_samples separate edge lists.
        Each edge list is a numpy array where:
        - For unweighted networks (UBCM): each row is [source_node, target_node]
        - For weighted networks (UECM): each row is [source_node, target_node, weight]

        If flatten=True: Single flattened list of all edges with network ID in last column:
        - For unweighted networks (UBCM): each row is [source_node, target_node, network_id]
        - For weighted networks (UECM): each row is [source_node, target_node, weight, network_id]

    Examples
    --------
    >>> import numpy as np
    >>> from fastmaxent import sampling

    # Sample unweighted network (UBCM)
    >>> alpha = np.array([1.5, 2.0, 1.0, 2.5])  # fitness parameters
    >>> edges = sampling(alpha, weighted=False, n_samples=1)
    >>> print(edges[0])  # First sample's edge list
    [[0 1]
     [1 2]
     [2 3]]

    # Sample degrees directly for faster computation
    >>> degrees = sampling(alpha, weighted=False, n_samples=1, return_degrees=True)
    >>> print(degrees[0])  # First sample's degree sequence
    [1 2 2 1]

    # Sample weighted network (UECM)
    >>> alpha = np.array([1.5, 2.0, 1.0, 2.5])
    >>> beta = np.array([0.5, 1.0, 0.8, 1.2])
    >>> edges = sampling(alpha, beta=beta, weighted=True, n_samples=2)
    >>> print(edges[0])  # First sample's edge list with weights
    [[0 1 3]
     [1 2 1]
     [2 3 2]]

    Raises
    ------
    AssertionError
        If beta is None when weighted=True, or if alpha and beta have different lengths,
        or if n_samples <= 0.

    Notes
    -----
    The sampling algorithms are optimized using Numba for fast execution. The UBCM
    sampler generates unweighted networks following the configuration model with
    given expected degrees. The UECM sampler generates weighted networks following
    the enhanced configuration model with given expected degrees and strengths.

    The algorithms use rejection sampling with geometric jumps to skip unlikely
    edge proposals, significantly improving sampling efficiency for sparse networks.

    Using return_degrees=True is recommended for parameter inference as it avoids
    the overhead of edge list to degree sequence conversion.
    """

    if weighted:
        assert (
            beta is not None
        ), "Beta is required for sampling weighted networks with Enhanced Configuration Model"
        assert len(alpha) == len(beta), "Alpha and beta must have the same length"

        sampler = _uecm_sampler
        parameters = {"alpha": alpha, "beta": beta, "n_samples": n_samples}
    else:
        sampler = _ubcm_sampler
        parameters = {"alpha": alpha, "n_samples": n_samples}

    assert n_samples > 0, "Number of samples must be greater than 0"

    # Generate all samples in a single call for better performance
    all_edges = sampler(**parameters)

    # Return flattened or separated edge lists based on flatten parameter
    if flatten:
        return all_edges

    # Return edge lists separated by network ID (original behavior)
    edge_list = []
    for sample_id in range(n_samples):
        sample_edges = []
        for edge in all_edges:
            if weighted:
                if edge[3] == sample_id:  # network_id is in position 3 for weighted
                    sample_edges.append(edge[:3])  # [source, target, weight]
            else:
                if edge[2] == sample_id:  # network_id is in position 2 for unweighted
                    sample_edges.append(edge[:2])  # [source, target]
        edge_list.append(sample_edges)
    return edge_list


########################################################################
######################### Unweighted case below ########################
########################################################################
@jit(nopython=True, nogil=True, fastmath=True)
def _ubcm_sampler(alpha, n_samples):
    """
    This function extends the application of Miller and Hagberg (MH) algorithm 2011
    to generate multiple undirected unweighted configuration model (UBCM) networks.
    Generate multiple unweighted graphs using the extended MH algorithm for UBCM.

    This function implements an extension of the Miller and Hagberg (MH) 2011
    algorithm to generate multiple undirected unweighted configuration model (UBCM)
    networks in a single call for improved performance.

    Parameters
    ----------
    alpha : numpy.ndarray
        An array of alpha values for each node, as defined in the manuscript.
    n_samples : int, default=1
        Number of network samples to generate.

    Returns
    -------
    list
        List of edges with network ID in last column [source, target, network_id]
    """

    edge_list = []
    n_nodes = len(alpha)

    permutation = np.argsort(alpha)
    sorted_alpha = alpha[permutation]

    for sample_id in range(n_samples):
        for i in range(n_nodes):
            j, q = i + 1, 1

            while j < n_nodes:
                a_i = sorted_alpha[i]
                a_j = sorted_alpha[j]
                p_ubcm = np.exp(-a_i - a_j) / (1 + np.exp(-a_i - a_j))

                if np.random.rand() < p_ubcm / q:
                    p_i = permutation[i]
                    p_j = permutation[j]
                    edge_list.append([p_i, p_j, sample_id])

                q = p_ubcm
                L = np.random.geometric(p=q)

                j += L

    return edge_list


########################################################################
######################### Weighted case below ##########################
########################################################################


@jit(nopython=True, nogil=True, fastmath=True)
def _find_min_two_beta_sum(beta_sorted, permutation_beta, mask):
    """
    Find the sum of the two smallest beta values among unprocessed nodes.

    This helper function is used in the UECM sampling algorithm to compute
    the minimum possible beta sum for the proposal distribution.

    Parameters
    ----------
    beta_sorted : numpy.ndarray
        Beta values sorted in ascending order.
    permutation_beta : numpy.ndarray
        Permutation indices that sort beta values.
    mask : numpy.ndarray of bool
        Boolean mask indicating which nodes have been processed (True)
        or are still available (False).

    Returns
    -------
    float
        Sum of the two smallest beta values among unprocessed nodes.
        Returns 0 if fewer than 2 nodes remain unprocessed.
    """
    cnt = 0
    retval = 0
    for i in range(len(beta_sorted)):
        if mask[permutation_beta[i]]:
            continue
        retval += beta_sorted[i]
        cnt += 1
        if cnt == 2:
            break

    return retval


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _uecm_sampler(alpha, beta, n_samples):
    """
    Implements a rejection sampling algorithm for the Undirected Enhanced
    Configuration Model (UECM) to generate multiple weighted networks.

    This function samples multiple weighted networks according to the UECM probability
    distribution using a rejection sampling approach. It processes nodes in order of
    increasing alpha+beta values and proposes edges with a proposal distribution.

    Parameters:
    -----------
    alpha : numpy.ndarray
        Array of alpha parameters for each node, controlling degree constraints.
    beta : numpy.ndarray
        Array of beta parameters for each node, controlling strength constraints.
    n_samples : int, default=1
        Number of network samples to generate.

    Returns:
    --------
    list
        List of sampled edges as [source, target, weight, network_id]
    """

    n_nodes = len(alpha)
    eps = 1e-2

    permutation_alpha_beta = np.argsort(alpha + beta)
    permutation_beta = np.argsort(beta)
    beta_sorted = beta[permutation_beta]

    edge_list = []

    for sample_id in range(n_samples):
        mask_beta = np.full(n_nodes, False)

        for _i in range(n_nodes):
            i = permutation_alpha_beta[_i]

            min_beta_sum = _find_min_two_beta_sum(
                beta_sorted, permutation_beta, mask_beta
            )

            q = np.exp(-alpha[i] - alpha[i] - beta[i] - beta[i]) / (
                1
                - np.exp(-min_beta_sum)
                + np.exp(-alpha[i] - alpha[i] - beta[i] - beta[i])
            )

            q_clipped = max(min(q, 1 - eps), eps)
            L = np.random.geometric(q_clipped)
            _j = _i + L

            while _j < n_nodes:

                j = permutation_alpha_beta[_j]
                p = np.exp(-alpha[i] - alpha[j] - beta[i] - beta[j]) / (
                    1
                    - np.exp(-beta[i] - beta[j])
                    + np.exp(-alpha[i] - alpha[j] - beta[i] - beta[j])
                )

                if np.random.rand() < p / q:
                    w = np.random.geometric(1 - np.exp(-beta[i] - beta[j]))
                    edge_list.append([i, j, w, sample_id])

                q = np.exp(-alpha[i] - alpha[j] - beta[i] - beta[j]) / (
                    1
                    - np.exp(-min_beta_sum)
                    + np.exp(-alpha[i] - alpha[j] - beta[i] - beta[j])
                )
                q_clipped = max(min(q, 1 - eps), eps)

                L = np.random.geometric(q_clipped)
                _j = _j + np.maximum(L, 1)

            mask_beta[i] = True

    return edge_list
