import math, numpy as np, networkx as nx, matplotlib.pyplot as plt, seaborn as sns
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import dijkstra

def spectral_gap(L, cutoff=0.05, return_ivals=False):
    """Implements the combination cutoff and gap heuristics discussed in the paper.
    
    Parameters
    ----------
    L : scipy sparse matrix
        n x n Laplacian matrix.
    cutoff : float
        Value of the cutoff used in the cutoff heuristic.
    return_ivals : boolean
        Function will also return a numpy array of eigenvalues if True.

    Returns
    -------
    number of clusters C, lambda_C - lambda_{C-1}, lambda_C 
    """
    ivals = eigh(L.todense(), eigvals_only=True)

    num_clusters = np.max(np.where(ivals<=cutoff)[0]) + 1
    gap_size = ivals[num_clusters] - ivals[num_clusters - 1]
    max_below = np.max(ivals[0:num_clusters]) # largest eigenvalue below the gap
        
    if return_ivals:
        return num_clusters, gap_size, max_below, ivals
    else:
        return num_clusters, gap_size, max_below

def spectral_clustering(num_clusters, L, seed):
    """Implements the 3rd spectral clustering algorithm listed in von Luxburg (2007), which she attributes to Ng, Jordan, and Weiss (2002).

    Parameters
    ----------
    num_clusters : int
        Number of desired clusters.
    L : scipy sparse matrix
        n x n Laplacian matrix.
    seed : int
        Seed for k-means clustering function. Set to None to not set seed.

    Returns
    -------
    n-dimensional numpy array of cluster labels from 0 to num_clusters-1. In the monte carlo, we apply it only to the giant component.
    """
    ivals, ivecs = eigsh(L, k=num_clusters, which='SM')
    ivecs /= np.sqrt( (ivecs**2).sum(axis=1) )[:,None] # row normalize by row norm
    kmeans = KMeans(num_clusters, n_init=30, random_state=seed).fit(ivecs)
    return kmeans.labels_

def conductance(clusters, A, A_giant):
    """Computes maximal conductance of a set of clusters.

    Parameters
    ----------
    clusters : numpy array
        vector of cluster labels only for nodes in the giant component (A_giant). For example, this can be the output of spectral_clustering() applied to the giant.
    A : NetworkX graph
    A_giant : NetworkX graph
        Giant component of A.

    Returns
    -------
    maximal conductance, vector of cluster labels for all nodes in A
    
    Nodes not in the giant component are assigned a label of -1.
    """
    conductances = np.zeros(int(np.max(clusters)))
    excluded = np.sort(list(set(A.nodes) - set(A_giant.nodes))) # nodes not in giant
    all_clusters = list(clusters).copy()
    for i in excluded: 
        # spectral_clustering() is used only on the giant, but now we want to add those not in the giant back in. We will assign these excluded nodes a cluster membership of -1, while making sure to keep the original labeling of the nodes.
        all_clusters = all_clusters[0:i] + [-1] + all_clusters[i:len(all_clusters)]
    all_clusters = np.array(all_clusters)
    for i in range(1,np.max(clusters)+1):
        S = np.where(all_clusters==i)[0]
        conductances[i-1] = nx.cut_size(A, S) / nx.volume(A, S)
    return np.max(conductances), all_clusters

def sample_means(Y, clusters):
    """Computes average of Y for each cluster.

    Parameters
    ----------
    Y : numpy array
        Vector of observations.
    clusters : numpy array
        Vector of cluster indicators, same dimension as Y.

    Returns
    -------
    numpy array of averages, one for each unique value in clusters
    """
    thetahat = []
    for C in np.unique(clusters):
        if C != -1:
            members = np.where(clusters==C)
            Y2 = Y[members]
            thetahat.append( Y2.mean() )
    if len(thetahat) == 1:
        thetahat = thetahat[0]
    else:
        thetahat = np.array(thetahat)
    return thetahat

def make_Zs(Y,ind1,ind0,pscores1,pscores0,subsample=False):
    """Returns vector of Z_i's, used to construct IPW estimator.

    Parameters
    ----------
    Y : numpy float array
        n-dimensional outcome vector.
    ind1 : numpy boolean array
        n-dimensional vector of indicators for first exposure mapping.
    ind0 : numpy boolean array
        n-dimensional vector of indicators for second exposure mapping.
    pscores1 : numpy float array
        n-dimensional vector of probabilities of first exposure mapping for each unit.
    pscores0 : numpy float array
        n-dimensional vector of probabilities of second exposure mapping for each unit
    subsample : numpy boolean array
        When set to an object that's not a numpy array, the function will define subsample to be an n-dimensional array of ones, whereby it is assumed that all n units are included in the population. Otherwise, it must be an boolean array of the same dimension as Z where True components indicate population inclusion.

    Returns
    -------
    n-dimensional numpy float array, where entries corresponding to the True entries of subsample are equal to the desired Z's, and entries corresponding to False subsample entries are set to -1000.
    """
    if type(subsample) != np.ndarray: subsample = np.ones(Y.size, dtype=bool)
    i1 = ind1[subsample]
    i0 = ind0[subsample]
    ps1 = pscores1[subsample]
    ps0 = pscores0[subsample]
    weight1 = i1.copy().astype('float')
    weight0 = i0.copy().astype('float')
    weight1[weight1 == 1] = i1[weight1 == 1] / ps1[weight1 == 1]
    weight0[weight0 == 1] = i0[weight0 == 1] / ps0[weight0 == 1]
    Z = np.ones(Y.size) * (-1000) # filler entries that won't be used
    Z[subsample] = Y[subsample] * (weight1 - weight0)
    return Z

def network_HAC(Z, A, K=0, subsample=False):
    """Computes network HAC variance estimator

    Parameters
    ----------
    Z : (n x k)-dimensional numpy float array
    A : NetworkX undirected graph
        Graph on n nodes. NOTE: Assumes nodes are labeled 0 through n-1, so that the data for node i is given by the ith component of each array in Zs.
    K : int
        We take the max of the bandwidth and 2*K.
    subsample : numpy boolean array
        When set to an object that's not a numpy array, the function will define subsample to be an n-dimensional array of ones, whereby it is assumed that all n units are included in the population. Otherwise, it must be an boolean array of the same dimension as Z where True components indicate population inclusion.

    Returns
    -------
    var : (k x k)-dimensional numpy float array
        HAC estimate of variance-covariance matrix.
    """
    n = Z.shape[0]
    if type(subsample) != np.ndarray: 
        subsample = np.ones(n, dtype=bool) # handle case where subsample is False

    # compute path distances
    G = nx.to_scipy_sparse_matrix(A, nodelist=range(n), format='csr')
    dist_matrix = dijkstra(csgraph=G, directed=False, unweighted=True)
    Gcc = [A.subgraph(c).copy() for c in sorted(nx.connected_components(A), key=len, reverse=True)]
    giant = [i for i in Gcc[0]] # set of nodes in giant component
    APL = dist_matrix[np.ix_(giant,giant)].sum() / len(giant) / (len(giant)-1) # average path length
    avg_deg = G.dot(np.ones(n)[:,None]).mean()
    
    # default bandwidth
    exp_nbhd = APL < 2 * np.log(avg_deg) / np.log(n)
    b = round(APL/2) if exp_nbhd else round(APL**(1/3)) # bandwidth
    b = max(2*K,b)

    weights = (dist_matrix <= b)[np.ix_(subsample,subsample)] # weight matrix
    Zsub = Z[subsample] - Z[subsample].mean() # demeaned data

    var = Zsub.T.dot(weights.dot(Zsub)) / subsample.sum()
    if Z.ndim == 1:
        if var <= 0:
            print("Variance estimator is not positive definite; using alternative.")
            if b < 0: b = round(APL/4) if exp_nbhd else round(APL**(1/3)) # rec bandwidth
            b_neighbors = dist_matrix <= b
            row_sums = b_neighbors.dot(np.ones(n))
            b_norm = b_neighbors / np.sqrt(row_sums)[:,None]
            weights = b_norm.dot(b_norm.T)[np.ix_(subsample,subsample)]
            var = Zsub.dot(weights.dot(Zsub)) / subsample.sum()
    else:
        var = Zsub.T.dot(weights.dot(Zsub)) / subsample.sum() 
        if np.any(np.linalg.eigvals(var) <= 0):
            print("Variance estimator is not positive definite; using correction.")
            w, v = np.linalg.eigh(var)
            var = v @ np.diag(np.maximum(w, 0.05)) @ v.T

    return var

def rand_test(Y, mu, alpha):
    """Implements randomization test of Canay et al. (2017).

    Parameters
    ----------
    Y : numpy array
        Vector of observations.
    mu : float
        Null hypothesis.
    alpha : float
        Significance level.

    Returns
    -------
    boolean that is True iff the test rejects
    """
    q = Y.size
    ph = np.ones(q).astype(int)
    G = [tuple(ph), tuple(ph*(-1))] # list of permutations
    for j in range(q-1):
        ph[j] = -1
        G += list(perm_unique(ph.tolist()))
    G = np.array(G)

    rand_dist = np.zeros(G.shape[0])
    Lambda_nor = Y - mu
    for g in range(G.shape[0]):
        rand_dist[g] = test_stat(G[g] * Lambda_nor)

    T_wald = test_stat(Lambda_nor)
    p_value = (rand_dist >= T_wald).mean()
    reject = p_value < alpha
    return reject

def test_stat(Lambda_nor):
    """Test statistic for rand_test().

    Parameters
    ----------
    Lambda_nor : numpy array
        Vector of normalized network moments (centered at the null and scaled up by root(n)), one for each cluster.

    Returns
    -------
    value of test statistic
    """
    return Lambda_nor.size * Lambda_nor.mean()**2 / np.power(Lambda_nor, 2).mean()

def rand_test_mult(Y, mu, alpha):
    """Implements randomization test of Canay et al. (2017) for a vector of null hypotheses.

    Parameters
    ----------
    Y : numpy array
        Vector of observations.
    mu : numpy array
        Vector of null hypotheses.
    alpha : float
        Significance level.

    Returns
    -------
    boolean that is True iff the test rejects
    """
    q = Y.size
    ph = np.ones(q).astype(int)
    G = [tuple(ph), tuple(ph*(-1))] # list of permutations
    for j in range(q-1):
        ph[j] = -1
        G += list(perm_unique(ph.tolist()))
    G = np.array(G)

    rand_dist = np.zeros((G.shape[0],mu.size))
    for g in range(G.shape[0]):
        rand_dist[g,:] = test_stat_mult(G[g], Y, mu)

    T_wald = test_stat_mult(np.ones(G[0].size), Y, mu)
    p_vals = (rand_dist >= T_wald).mean(axis=0)
    reject = p_vals < alpha
    return reject

def test_stat_mult(pi, Y, mu):
    """Test statistic for rand_test_mult().

    Parameters
    ----------
    pi : numpy array
        Vector of +/-1 to generate randomization distribution.
    Y : numpy array
        Vector of data.
    mu : numpy array
        Vector of null hypotheses.

    Returns
    -------
    vector of test statistics
    """
    Lambda_nor = (Y[:,np.newaxis] - mu) * pi[:,np.newaxis]
    return Lambda_nor.shape[0] * np.power(Lambda_nor.mean(axis=0),2) / np.power(Lambda_nor, 2).mean(axis=0)

# this class and two subsequent functions are for perm_unique()
class unique_element:
    # object with two attributes
    # value is the number itself and occurrences is how many times the number appears in the raw data
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1

