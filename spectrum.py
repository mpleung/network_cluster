import numpy as np, pandas as pd, networkx as nx, matplotlib.pyplot as plt, seaborn as sns
from scipy.sparse import csgraph
from scipy.linalg import eigh
from inference_module import *
from DGP_module import *
from scipy.spatial.distance import pdist, squareform

B = 10000
n = 1000
avg_deg = 5
cutoff = 0.05

make_plot = True
if make_plot: 
    import os
    if not os.path.isdir('results'): os.mkdir('results')
    B = 1

gap_sizes = np.zeros((B,4))
num_clusters = np.zeros((B,4))
max_ival_below = np.zeros((B,4))
max_cluster_sizes = np.zeros((B,4))
med_cluster_sizes = np.zeros((B,4))
sd_cluster_sizes = np.zeros((B,4))
cluster_size_gaps = np.zeros((B,4))
conductances = np.zeros((B,4))
giant_size = np.zeros((B,4))
avg_degs = np.zeros((B,4))

for b in range(B):
    if make_plot:
        np.random.seed(seed=1)
    else:
        np.random.seed(seed=b)
    print('b = {}'.format(b))

    for i in range(4):
        print('  i = {}'.format(i))
        if i == 0: # random geometric graph 
            positions = np.random.uniform(size=(n,2))
            r_n = (avg_deg/ball_vol(2,1)/n)**(1/2)
            A = gen_RGG(positions, r_n)
        elif i==1: # random connections model
            alpha = np.random.uniform(size=n)
            r_n = (avg_deg/3.5/ball_vol(2,1)/n)**(1/2)
            positions = np.random.uniform(size=(n,2))
            latent_index = alpha + alpha[:,None] - squareform(pdist(positions / r_n))
            P_LSM = np.exp(latent_index) / (1 + np.exp(latent_index)) # n x n matrix of link probabilities
            np.fill_diagonal(P_LSM,0) # zero diagonals
            U = np.random.uniform(0,1,size=(n,n))
            U = np.tril(U) + np.tril(U, -1).T
            A = nx.from_numpy_matrix((U < P_LSM).astype(int))
        elif i==2: # erdos-renyi graph
            A = nx.fast_gnp_random_graph(n, avg_deg/n, seed=b)
        else: # stochastic block model
            K = 10 # number of blocks
            P_SBM = np.ones((n,n))*(avg_deg*4/9)/n
            bs = int(n/K)
            for r in range(K): P_SBM[(r*bs):((r+1)*bs),(r*bs):((r+1)*bs)] = 8*avg_deg/n
            np.fill_diagonal(P_SBM,0) # zero diagonals
            U = np.random.uniform(0,1,size=(n,n))
            U = np.tril(U) + np.tril(U, -1).T
            A = nx.from_numpy_matrix((U < P_SBM).astype(int))

        avg_degs[b,i] = np.array([A.degree[i] for i in A.nodes]).mean()
        A_components = [A.subgraph(c).copy() for c in sorted(nx.connected_components(A), key=len, reverse=True)]
        A_giant = A_components[0] # extract giant component
        giant_size[b,i] = A_giant.number_of_nodes()
        L = csgraph.laplacian(nx.to_scipy_sparse_matrix(A_giant, format="csr"), normed=True)
        num_clusters[b,i], gap_sizes[b,i], max_ival_below[b,i] = spectral_gap(L, cutoff=cutoff)
        clusters = spectral_clustering(int(num_clusters[b,i]), L, b) + 1
        cluster_sizes = np.sort(np.histogram(clusters, np.max(clusters))[0])
        max_cluster_sizes[b,i] = np.max(cluster_sizes)
        med_cluster_sizes[b,i] = np.median(cluster_sizes)
        sd_cluster_sizes[b,i] = cluster_sizes.std()
        if num_clusters[b,i] == 1:
            cluster_size_gaps[b,i] = cluster_sizes[0]
        else:
            cluster_size_gaps[b,i] = cluster_sizes[cluster_sizes.size-1] - cluster_sizes[cluster_sizes.size-2]
        conductances[b,i], all_clusters = conductance(clusters, A, A_giant)

        # plot networks and color by cluster
        if b==0 and i in [0,1] and make_plot:
            print('i: {}. Conductance: {}'.format(i,conductances[b,i]))
            if i==0:
                all_clusters = all_clusters * 2
            elif i==1: 
                all_clusters *= 6
            count = np.max(all_clusters) + 2
            for k,comp in enumerate(A_components[1:]):
                all_clusters[list(comp)] = count
                count += 1
            pos = dict(zip(list(A), positions[list(A),:].tolist()))
            plt.figure(figsize=(8, 8))
            nx.draw_networkx_edges(A, pos, alpha=0.4)
            nx.draw_networkx_nodes(A, pos, node_size=80, edgecolors='black', node_color=all_clusters, cmap=plt.cm.tab20b)
            if i==0: 
                plt.axis("off")
                plt.tight_layout()
                plt.savefig('results/RGG_clusters.png', dpi=200)
            else:
                plt.axis("off")
                plt.tight_layout()
                plt.savefig('results/RCM_clusters.png', dpi=200)

# produce table
if not make_plot:
    table = pd.DataFrame(np.vstack([ conductances.mean(axis=0), num_clusters.mean(axis=0), gap_sizes.mean(axis=0), max_ival_below.mean(axis=0), med_cluster_sizes.mean(axis=0), giant_size.mean(axis=0), avg_degs.mean(axis=0) ]).T)
    table.index = ['RGG','RCM','ER','SBM']
    table.columns = ['$\max_\ell \phi(\C_\ell)$', '\# Clus.', 'Gap', '$\\lambda_L$', 'Med.\ Clus.', 'Giant', 'Degree']
    print('\n\\begin{table}[ht]')
    print('\centering')
    print('\caption{Spectra and Clusters}')
    print('\\begin{threeparttable}')
    print(table.to_latex(float_format = lambda x: '%.3f' % x, header=True, escape=False, multicolumn_format='c'))
    print('\\begin{tablenotes}[para,flushleft]')
    print("  \\footnotesize $n={}$. Averages over {} simulations. ``Gap'' = size of spectral gap, ``Med.\ Clus.'' = median cluster size, ``Giant'' = size of giant component, ``Degree'' = average degree.".format(B,n))
    print('\end{tablenotes}')
    print('\end{threeparttable}')
    print('\end{table}')

# plot spectrum
if make_plot:
    np.random.seed(seed=1)

    positions_RGG = np.random.uniform(size=(n,2))
    A_RGG = gen_RGG(positions_RGG, (avg_deg/ball_vol(2,1)/n)**(1/2))
    A_RGG_giant = [A_RGG.subgraph(c).copy() for c in sorted(nx.connected_components(A_RGG), key=len, reverse=True)][0] # extract giant component
    L_RGG = csgraph.laplacian(nx.to_scipy_sparse_matrix(A_RGG_giant, format="csr"), normed=True)
    ivals_RGG = eigh(L_RGG.todense(), eigvals_only=True)

    A_ER = nx.fast_gnp_random_graph(n, avg_deg/n, seed=0) # seed
    A_ER_giant = [A_ER.subgraph(c).copy() for c in sorted(nx.connected_components(A_ER), key=len, reverse=True)][0] # extract giant component
    L_ER = csgraph.laplacian(nx.to_scipy_sparse_matrix(A_ER_giant, format="csr"), normed=True)
    ivals_ER = eigh(L_ER.todense(), eigvals_only=True)

    alpha = np.random.uniform(size=n)
    r_n = (avg_deg/3/ball_vol(2,1)/n)**(1/2)
    positions_RCM = np.random.uniform(size=(n,2))
    latent_index = alpha + alpha[:,None] - squareform(pdist(positions_RCM / r_n))
    P_LSM = np.exp(latent_index) / (1 + np.exp(latent_index)) # n x n matrix of link probabilities
    np.fill_diagonal(P_LSM,0) # zero diagonals
    U = np.random.uniform(0,1,size=(n,n))
    U = np.tril(U) + np.tril(U, -1).T
    A_RCM = nx.from_numpy_matrix((U < P_LSM).astype(int))
    A_RCM_giant = [A_RCM.subgraph(c).copy() for c in sorted(nx.connected_components(A_RCM), key=len, reverse=True)][0] # extract giant component
    L_RCM = csgraph.laplacian(nx.to_scipy_sparse_matrix(A_RCM_giant, format="csr"), normed=True)
    ivals_RCM = eigh(L_RCM.todense(), eigvals_only=True)

    K = 10 # number of blocks
    P_SBM = np.ones((n,n))*(avg_deg*4/9)/n # n x n matrix of link probabilities
    bs = int(n/K)
    for r in range(K): P_SBM[(r*bs):((r+1)*bs),(r*bs):((r+1)*bs)] = 8*avg_deg/n
    np.fill_diagonal(P_SBM,0) # zero diagonals
    U = np.random.uniform(0,1,size=(n,n))
    U = np.tril(U) + np.tril(U, -1).T
    A_SBM = nx.from_numpy_matrix((U < P_SBM).astype(int))
    A_SBM_giant = [A_SBM.subgraph(c).copy() for c in sorted(nx.connected_components(A_SBM), key=len, reverse=True)][0] # extract giant component
    L_SBM = csgraph.laplacian(nx.to_scipy_sparse_matrix(A_SBM_giant, format="csr"), normed=True)
    ivals_SBM = eigh(L_SBM.todense(), eigvals_only=True)

    sns.set_theme(style='dark', font='candara')
    eps = 0.05
    fig, axes = plt.subplots(2, 4, figsize=(10, 4))
    axes[0,0].set(ylim=(0,130))
    axes[0,0].set_yticks([0,50,100])
    axes[0,0].set(xlim=(-0.1,2.05))
    axes[0,0].set_xticks([0,1,2])
    axes[0,0].set_title('RGG histogram')
    sns.set_style('dark')
    sns.histplot(data=ivals_RGG, ax=axes[0,0])
    axes[0,1].set_title('RGG scatterplot')
    axes[0,1].set(ylabel='Eigenvalues')
    axes[0,1].set(ylim=(0-eps*2,2+eps))
    axes[0,1].set_yticks([0,1,2])
    axes[0,1].set(xlim=(-30,1030))
    axes[0,1].set_xticks([0,500,1000])
    sns.scatterplot(x=np.arange(len(ivals_RGG)), y=ivals_RGG, linewidth=0, s=10, ax=axes[0,1])
    axes[1,0].set(ylim=(0,130))
    axes[1,0].set_yticks([0,50,100])
    axes[1,0].set(xlim=(-0.1,2.05))
    axes[1,0].set_xticks([0,1,2])
    axes[1,0].set_title('ER histogram')
    sns.histplot(data=ivals_ER, ax=axes[1,0])
    axes[1,1].set_title('ER scatterplot')
    axes[1,1].set(ylabel='Eigenvalues')
    axes[1,1].set(ylim=(0-eps*2,2+eps))
    axes[1,1].set_yticks([0,1,2])
    axes[1,1].set(xlim=(-30,1030))
    axes[1,1].set_xticks([0,500,1000])
    sns.scatterplot(x=np.arange(len(ivals_ER)), y=ivals_ER, linewidth=0, s=10, ax=axes[1,1])
    axes[0,2].set(ylim=(0,130))
    axes[0,2].set_yticks([0,50,100])
    axes[0,2].set(xlim=(-0.1,2.05))
    axes[0,2].set_xticks([0,1,2])
    axes[0,2].set_title('RCM histogram')
    sns.histplot(data=ivals_RCM, ax=axes[0,2])
    axes[0,3].set_title('RCM scatterplot')
    axes[0,3].set(ylabel='Eigenvalues')
    axes[0,3].set(ylim=(0-eps*2,2+eps))
    axes[0,3].set_yticks([0,1,2])
    axes[0,3].set(xlim=(-30,1030))
    axes[0,3].set_xticks([0,500,1000])
    sns.scatterplot(x=np.arange(len(ivals_RCM)), y=ivals_RCM, linewidth=0, s=10, ax=axes[0,3])
    axes[1,2].set(ylim=(0,130))
    axes[1,2].set_yticks([0,50,100])
    axes[1,2].set(xlim=(-0.1,2.05))
    axes[1,2].set_xticks([0,1,2])
    axes[1,2].set_title('SBM histogram')
    sns.histplot(data=ivals_SBM, ax=axes[1,2])
    axes[1,3].set_title('SBM scatterplot')
    axes[1,3].set(ylabel='Eigenvalues')
    axes[1,3].set(ylim=(0-eps*2,2+eps))
    axes[1,3].set_yticks([0,1,2])
    axes[1,3].set(xlim=(-30,1030))
    axes[1,3].set_xticks([0,500,1000])
    sns.scatterplot(x=np.arange(len(ivals_SBM)), y=ivals_SBM, linewidth=0, s=10, ax=axes[1,3])
    plt.tight_layout()
    plt.savefig('results/sim_spectra.png',bbox_inches='tight',dpi=200)

