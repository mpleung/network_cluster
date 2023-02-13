import numpy as np, pandas as pd, networkx as nx, os
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from scipy.sparse import csgraph
from inference_module import *
from DGP_module import *

if not os.path.isdir('results'): os.mkdir('results')

B = 10000 # number of sims
Ns = [250, 500] # network sizes
nclus = 8 # number of clusters desired in the giant component
kappa = 5 # avg degree
sigf_lvl = 0.05
model = 'RGG' # network formation model. options: RGG, RCM

# storage
alternatives = np.linspace(0,0.75,101)[1:]
test_rand = np.zeros((len(Ns),B,alternatives.size))
ttest_HAC = np.zeros((len(Ns),B,alternatives.size))

for q,n in enumerate(Ns):
    for b in range(B):
        seed = int(n*(B*10) + b)
        np.random.seed(seed=seed)
        print('b = {}'.format(b))

        errors = np.random.normal(0,1,size=n)
        
        # simulate network
        if model=='RGG': # random geometric graph 
            positions = np.random.uniform(size=(n,2))
            r_n = (kappa/ball_vol(2,1)/n)**(1/2)
            A = gen_RGG(positions, r_n)
        else: # random connections model
            positions = np.random.uniform(size=(n,2))
            alpha = np.random.uniform(size=n)
            r_n = (kappa/3.5/ball_vol(2,1)/n)**(1/2)
            latent_index = alpha + alpha[:,None] - squareform(pdist(positions / r_n))
            P_LSM = np.exp(latent_index) / (1 + np.exp(latent_index)) # n x n matrix of link probabilities
            np.fill_diagonal(P_LSM,0) # zero diagonals
            U = np.random.uniform(0,1,size=(n,n))
            U = np.tril(U) + np.tril(U, -1).T
            A = nx.from_numpy_matrix((U < P_LSM).astype(int))
        A_mat = nx.to_scipy_sparse_matrix(A, nodelist=range(n), format='csc')

        # simulate outcomes
        Y, A_norm = gen_Y(A_mat, errors) 

        # construct clusters
        A_components = [A.subgraph(c).copy() for c in sorted(nx.connected_components(A), key=len, reverse=True)]
        A_giant = A_components[0] # extract giant component
        L = csgraph.laplacian(nx.to_scipy_sparse_matrix(A_giant, format="csr"), normed=True)
        clusters = spectral_clustering(nclus, L, b) + 1
        _, clusters = conductance(clusters, A, A_giant)
        for C in A_components[1:len(A_components)]:
            if C.number_of_nodes() >= 20:
                clusters[list(C)] = np.max(clusters) + 1 # assign cluster labels to nontrivial components
            else:
                clusters[list(C)] = -1 # trivial components have label -1
        cluster_sizes = np.sort(np.histogram(clusters, np.max(clusters))[0])

        # sample means
        thetahat_clusters = sample_means(Y, clusters)

        # randomization test
        test_rand[q,b,:] = rand_test_mult(thetahat_clusters, alternatives, sigf_lvl)

        # t-test using HAC estimator
        thetahat = Y.mean()
        HAC_SE = np.sqrt(network_HAC(Y, A) / n)
        ttest_HAC[q,b,:] = np.abs( (thetahat-alternatives) / HAC_SE ) > norm.ppf(1-sigf_lvl/2)

# save results in csv
results = pd.DataFrame(np.vstack([ test_rand[0,:,:].mean(axis=0), test_rand[1,:,:].mean(axis=0), ttest_HAC[0,:,:].mean(axis=0), ttest_HAC[1,:,:].mean(axis=0) ]).T)
results.columns = ['rand250', 'rand500', 'hac250', 'hac500']
results.to_csv('results/results_power_' + model + '.csv', float_format='%.4f', index=False)

