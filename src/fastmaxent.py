import numpy as np
from numba import njit, jit

########################################################################
######################### Unweighted case below ########################
########################################################################
@jit(nopython=True, nogil=True, fastmath=True, cache=True, parallel=True)
def unweighted_extended_mh_cl_model(alpha):
    """
    This function extends the application of Miller and Hagberg (MH) algorithm 2011
    to generate undirected unweighted configuration model (UBCM) from algorithm.
    Generate an unweighted graph using the extended MH algorithm for UBCM.

    This function implements an extension of the Miller and Hagberg (MH) 2011
    algorithm to generate an undirected unweighted configuration model (UBCM).

    Parameters
    ----------
    alpha : numpy.ndarray
        An array of alpha values for each node, as defined in the manuscript.

    Returns
    -------
    numpy.ndarray
        An array representing the edge list of the generated graph.
    """

    edge_list = []
    n_nodes = len(alpha)

    permutation = np.argsort(alpha)
    sorted_alpha = alpha[permutation]

    for i in range(n_nodes):
        j, q = i+1, 1

        while j < n_nodes:
            a_i = sorted_alpha[i]
            a_j = sorted_alpha[j]
            p_ubcm = np.exp(-a_i - a_j) / (1 + np.exp(-a_i - a_j))  

            if np.random.rand() < p_ubcm / q:
                p_i = permutation[i]
                p_j = permutation[j]
                edge_list.append([p_i, p_j])

            q = p_ubcm
            L = np.random.geometric(p=q)

            j += L

    edge_list = np.array(edge_list)

    return edge_list



########################################################################
######################### Weighted case below ##########################
########################################################################

@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def find_min_two_beta_sum(beta_sorted, permutation_beta, mask):
    """
    Find the minimum beta value among unprocessed nodes.

    Parameters:
    -----------
    beta_sorted : numpy array
        Beta values sorted in ascending order
    permutation_beta : numpy array
        Permutation of beta values
    mask : numpy array of bool
        Indicates which nodes have been processed

    Returns:
    --------
    retval: Minimum beta value
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

@jit(nopython=True, nogil=True, fastmath=True, cache=True, parallel=True)
def weighted_extended_mh_cl_model(alpha, beta):
    """
    Implements a rejection sampling algorithm for the Undirected Enhanced 
    Configuration Model (UECM).

    This function samples weighted edges according to the UECM probability distribution
    using a rejection sampling approach. It processes nodes in order of increasing
    alpha+beta values and proposes edges with a proposal distribution.

    Parameters:
    -----------
    alpha : numpy.ndarray
        Array of alpha parameters for each node, controlling degree constraints.
    beta : numpy.ndarray
        Array of beta parameters for each node, controlling strength constraints.

    Returns:
    --------
    edge_list : list of tuples
        List of sampled edges as (source, target, weight) tuples.
    n_accepted : int
        Number of accepted edge proposals.
    n_proposed : int
        Total number of proposed edges.
    """

    n_nodes = len(alpha)
    eps = 1e-2       

    permutation_alpha_beta = np.argsort(alpha + beta)
    permutation_beta = np.argsort(beta)
    beta_sorted = beta[permutation_beta]

    mask_beta = np.full(n_nodes, False) 

    n_proposed, n_accepted = 0, 0
    edge_list = []

    for _i in range(n_nodes):
        i = permutation_alpha_beta[_i] 

        min_beta_sum = find_min_two_beta_sum(beta_sorted, permutation_beta, mask_beta)

        q = np.exp(-alpha[i] - alpha[i] - beta[i] - beta[i]) / (1 - np.exp(-min_beta_sum) + 
                    np.exp(-alpha[i] - alpha[i] - beta[i] - beta[i]))

        q_clipped = max(min(q, 1 - eps), eps) 
        L = np.random.geometric(q_clipped)
        _j = _i + L

        while _j < n_nodes:

            j = permutation_alpha_beta[_j]
            p = np.exp(-alpha[i] - alpha[j] - beta[i] - beta[j]) / (1 -         
                        np.exp(-beta[i] - beta[j]) +
                        np.exp(-alpha[i] - alpha[j] - beta[i] - beta[j]))

            n_proposed += 1
            if np.random.rand() < p / q:
                w = np.random.geometric(1 - np.exp(-beta[i] - beta[j]))
                edge_list.append([i, j, w])
                n_accepted += 1

            q = np.exp(-alpha[i] - alpha[j] - beta[i] - beta[j]) / (1 - np.exp(-min_beta_sum) +
                    np.exp(-alpha[i] - alpha[j] - beta[i] - beta[j]))
            q_clipped = max(min(q, 1 - eps), eps)

            L = np.random.geometric(q_clipped)
            _j = _j + np.maximum(L, 1)

        mask_beta[i] = True                                                  

    edge_list = np.array(edge_list)

    return edge_list, n_accepted, n_proposed