import numpy as np, pandas as pd, networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from scipy.sparse import csgraph
from inference_module import *
from DGP_module import *

B = 10000 # number of sims
Ns = [250, 500, 1000] # network sizes
nclus = 8 # number of clusters desired in the giant component
kappa = 5 # avg degree
sigf_lvl = 0.05

# storage
phi = np.zeros((B,len(Ns),3)) # conductance
actual_num_clusters = np.zeros((B,len(Ns),3)) # number of clusters including other components
cluster_size_1 = np.zeros((B,len(Ns),3)) # size of largest cluster
cluster_size_2 = np.zeros((B,len(Ns),3)) # size of 2nd largest
cluster_size_l = np.zeros((B,len(Ns),3)) # size of smallest with at least 20 nodes
test_rand = np.zeros((B,len(Ns),3)) # randomization test
ttest_HAC = np.zeros((B,len(Ns),3)) # t-test with HAC
ttest_naive = np.zeros((B,len(Ns),3)) # t-test with iid SEs
estimator_RGG = np.zeros(B)
estimator_ER = np.zeros(B)

for q,n in enumerate(Ns):
    for b in range(B):
        seed = int(n*(B*10) + b)
        np.random.seed(seed=seed)
        print('b = {}'.format(b))

        errors = np.random.normal(0,1,size=n)
        
        for i in range(3):
            print('  i = {}'.format(i))

            # simulate network
            if i == 0: # random geometric graph 
                positions = np.random.uniform(size=(n,2))
                r_n = (kappa/ball_vol(2,1)/n)**(1/2)
                A = gen_RGG(positions, r_n)
            elif i == 1: # random connections model
                positions = np.random.uniform(size=(n,2))
                alpha = np.random.uniform(size=n)
                r_n = (kappa/3.5/ball_vol(2,1)/n)**(1/2)
                latent_index = alpha + alpha[:,None] - squareform(pdist(positions / r_n))
                P_LSM = np.exp(latent_index) / (1 + np.exp(latent_index)) # n x n matrix of link probabilities
                np.fill_diagonal(P_LSM,0) # zero diagonals
                U = np.random.uniform(0,1,size=(n,n))
                U = np.tril(U) + np.tril(U, -1).T
                A = nx.from_numpy_matrix((U < P_LSM).astype(int))
            else: # SBM
                K = 10 # number of blocks
                P_SBM = np.ones((n,n))*(kappa*4/9)/n # n x n matrix of link probabilities
                bs = int(n/K)
                for r in range(K): P_SBM[(r*bs):((r+1)*bs),(r*bs):((r+1)*bs)] = 8*kappa/n
                np.fill_diagonal(P_SBM,0) # zero diagonals
                U = np.random.uniform(0,1,size=(n,n))
                U = np.tril(U) + np.tril(U, -1).T
                A = nx.from_numpy_matrix((U < P_SBM).astype(int))
            A_mat = nx.to_scipy_sparse_matrix(A, nodelist=range(n), format='csc')

            # simulate outcomes
            Y, A_norm = gen_Y(A_mat, errors) 

            # construct clusters
            A_components = [A.subgraph(c).copy() for c in sorted(nx.connected_components(A), key=len, reverse=True)]
            A_giant = A_components[0] # extract giant component
            L = csgraph.laplacian(nx.to_scipy_sparse_matrix(A_giant, format="csr"), normed=True)
            clusters = spectral_clustering(nclus, L, b) + 1
            phi[b,q,i], clusters = conductance(clusters, A, A_giant)
            for C in A_components[1:len(A_components)]:
                if C.number_of_nodes() >= 20:
                    clusters[list(C)] = np.max(clusters) + 1 # assign cluster labels to nontrivial components
                else:
                    clusters[list(C)] = -1 # trivial components have label -1
            actual_num_clusters[b,q,i] = np.unique(clusters).size - 1
            cluster_sizes = np.sort(np.histogram(clusters, np.max(clusters))[0])
            cluster_size_1[b,q,i] = cluster_sizes[cluster_sizes.size-1]
            cluster_size_2[b,q,i] = cluster_sizes[cluster_sizes.size-2]
            cluster_size_l[b,q,i] = cluster_sizes[cluster_sizes >= 20][0]

            # sample means
            thetahat = sample_means(Y, clusters)

            # randomization test
            test_rand[b,q,i] = rand_test(thetahat, 0, sigf_lvl)

            # t-test using HAC estimator
            thetahat = Y.mean()
            if i==0:
                estimator_RGG[b] = thetahat
            else:
                estimator_ER[b] = thetahat
            HAC_SE = np.sqrt(network_HAC(Y, A) / n)
            ttest_HAC[b,q,i] = np.abs( thetahat / HAC_SE ) > norm.ppf(1-sigf_lvl/2)

            # t-test using iid SEs
            iid_SE = np.sqrt(Y.var() / n)
            ttest_naive[b,q,i] = np.abs( thetahat / iid_SE ) > norm.ppf(1-sigf_lvl/2)

print('Estimates: {},{}'.format(estimator_RGG.mean(axis=0), estimator_ER.mean(axis=0)))

test_rand = test_rand.mean(axis=0)
ttest_HAC = ttest_HAC.mean(axis=0)
ttest_naive = ttest_naive.mean(axis=0)
phi = phi.mean(axis=0)
actual_num_clusters = actual_num_clusters.mean(axis=0)
cluster_size_1 = cluster_size_1.mean(axis=0)
cluster_size_2 = cluster_size_2.mean(axis=0)
cluster_size_l = cluster_size_l.mean(axis=0)

# print table
table = pd.DataFrame( np.hstack( [np.vstack( [test_rand[:,i], ttest_HAC[:,i], ttest_naive[:,i], phi[:,i], actual_num_clusters[:,i], cluster_size_1[:,i], cluster_size_2[:,i], cluster_size_l[:,i]] ) for i in range(test_rand.shape[1])] ) )
table.index = ['Rand', 'HAC', 'IID', '$\max_\ell \phi(\C_\ell)$', '\# Clusters', '1st Clus.', '2nd Clus.', 'Last Clus.']
table.columns = pd.MultiIndex.from_product([['RGG', 'RCM', 'SBM'], Ns])
print('\n\\begin{table}[ht]')
print('\centering')
print('\caption{Size Under Design 1}')
print('\\begin{threeparttable}')
print(table.to_latex(float_format = lambda x: '%.3f' % x, header=True, escape=False, multicolumn_format='c'))
print('\\begin{tablenotes}[para,flushleft]')
print("  \\footnotesize Averages over {} simulations. The first three rows report the sizes of level-5\% tests. The last three rows report cluster sizes in descending order of size.".format(B))
print('\end{tablenotes}')
print('\end{threeparttable}')
print('\end{table}')

