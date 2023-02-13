import numpy as np, pandas as pd, networkx as nx, matplotlib.pyplot as plt, seaborn as sns, sys, os
from scipy.sparse import csgraph
from scipy.sparse import spdiags, linalg
from scipy.linalg import eigh
from inference_module import *

path = 'results'
if not os.path.isdir(path): os.mkdir(path)
os.chdir('results')

make_plot = True

#### Load data ####

path = 'zacchia'
clusters = pd.read_csv('../' + path + '/Data/Networks Project/BSV/Gephi/Modularity Partitions/GephiMod06.csv').to_numpy()
clusters[:,1] += 1
edges = pd.read_stata('../' + path + '/Data/Networks Project/BSV/BSV Distance Metrics.dta', columns=['cusip_1','cusip_2','year','distance','sr_symm_invcor'])
edges = edges[edges.distance==1]
edges = edges.groupby(['cusip_1', 'cusip_2']).agg({'sr_symm_invcor': ['sum']})
edges.columns = ['weight']
edges = edges.reset_index()
edges = edges.to_numpy()

A = nx.Graph()
for i in range(len(edges)):
    A.add_edge(edges[i,0],edges[i,1],weight=edges[i,2])

excluded = list(set(A) - set(clusters[:,0]))
for i in excluded:
    A.remove_node(i)

n = A.number_of_nodes()
APL = nx.average_shortest_path_length(A)
diam = nx.diameter(A)
avg_deg = np.array([A.degree(i) for i in A]).mean()
clustering = nx.average_clustering(A)
print('{} nodes, APL: {}, diameter: {}, avg deg: {}, clustering: {}\n'.format(n,APL,diam,avg_deg,clustering))

#### Analysis with original weighted network ####

M = nx.to_scipy_sparse_matrix(A, nodelist=clusters[:,0], weight='weight')
L_w = csgraph.laplacian(nx.to_scipy_sparse_matrix(A, format="csr"), normed=True)
ivals = eigh(L_w.todense(), eigvals_only=True)
print('Spectrum of weighted Laplacian: {}\n'.format(ivals[1:6]))

conductances = np.zeros(int(np.max(clusters[:,1])))
cluster_sizes = np.zeros(int(np.max(clusters[:,1])))
for i in range(1,np.max(clusters[:,1])+1):
    S = np.where(clusters==i)[0]
    conductances[i-1] = M[S,:][:,list(set(range(n))-set(S))].sum() / M[S,:].sum()
    cluster_sizes[i-1] = S.size
print('Conductances of Louvain clusters (weighted graph): {}'.format(conductances))
print('Cluster sizes: {}\n'.format(cluster_sizes))
if make_plot:
    plot_data = pd.DataFrame(np.vstack([conductances, cluster_sizes]).T, columns=['Louvain Cond','Louvain Sizes'])
    plot_data.sort_values('Louvain Cond', inplace=True, ignore_index=True)

for q in [20, 5, 3]:
    _, ivecs = eigsh(L_w, k=q, which='SM')
    ivecs /= np.sqrt( (ivecs**2).sum(axis=1) )[:,None] # row normalize by row norm
    kmeans = KMeans(q, n_init=30, random_state=q).fit(ivecs)
    spec_clusters = kmeans.labels_ + 1
    spec_conductances = np.zeros(int(np.max(spec_clusters)))
    spec_cluster_sizes = np.zeros(int(np.max(spec_clusters)))
    for i in range(1,np.max(spec_clusters)+1):
        S = np.where(spec_clusters==i)[0]
        spec_conductances[i-1] = M[S,:][:,list(set(range(n))-set(S))].sum() / M[S,:].sum()
        spec_cluster_sizes[i-1] = S.size
    print('Conductances of {} spectral clusters (weighted graph): {}'.format(q,spec_conductances))
    print('Cluster sizes: {}\n'.format(spec_cluster_sizes))
    if make_plot:
        tmp = pd.DataFrame(np.vstack([spec_conductances, spec_cluster_sizes]).T, columns=['Spectral Cond','Spectral Sizes'])
        tmp.sort_values('Spectral Cond', inplace=True, ignore_index=True)
        tmp = pd.DataFrame(np.vstack([tmp.values, (-100)*np.ones((20-q,2))]), columns=['Spectral Cond ' + str(q),'Spectral Sizes ' + str(q)])
        plot_data = pd.concat([plot_data, tmp], axis=1)

if make_plot:
    plot_data['Labels'] = range(1,21)

    sns.set_theme(style='whitegrid', font='candara')
    g = sns.PairGrid(plot_data, x_vars=plot_data.columns[:-1], y_vars=['Labels'], height=6, aspect=.25)
    g.map(sns.stripplot, size=10, orient="h", palette="crest_r", linewidth=1, edgecolor="w")
    titles = ['Louvain', 'Louvain', 'Spectral (L=20)', 'Spectral (L=20)', 'Spectral (L=5)', 'Spectral (L=5)', 'Spectral (L=3)', 'Spectral (L=3)']
    xlabels = ['Conductance', 'Size', 'Conductance', 'Size', 'Conductance', 'Size', 'Conductance', 'Size']
    ylabels = ['Cluster Label', '', '', '', '', '', '', '']
    eps, eps1 = 0.07, 0.08
    xticks = [[0-eps,0.5,1+eps1], [plot_data.iloc[:,1].min(),(plot_data.iloc[:,1].max()+plot_data.iloc[:,1].min())/2,plot_data.iloc[:,1].max()], [0-eps,0.5,1+eps1], [plot_data.iloc[:,3].min(),(plot_data.iloc[:,3].max()+plot_data.iloc[:,3].min())/2,plot_data.iloc[:,3].max()], [0-eps,0.5,1+eps1], [plot_data.iloc[0:5,5].min(),(plot_data.iloc[:,5].max()+plot_data.iloc[0:5,5].min())/2,plot_data.iloc[:,5].max()], [0-eps,0.5,1+eps1], [plot_data.iloc[0:3,7].min(),(plot_data.iloc[:,7].max()+plot_data.iloc[0:3,7].min())/2,plot_data.iloc[:,7].max()]]
    count = 0
    for ax, title, xlabel, ylabel, xtick in zip(g.axes.flat, titles, xlabels, ylabels, xticks):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax.set_xticks(xtick)
        if count % 2 == 0:
            ax.set_xticklabels([0,0.5,1])
            ax.set(xlim=(xtick[0],xtick[2]))
        else:
            ax.set_xticklabels([int(round(i)) for i in xtick])
        if count in [5,7]:
            ax.set(xlim=(xtick[0]-30,xtick[2]+30))
        elif count == 3:
            ax.set(xlim=(xtick[0]-5,xtick[2]+5))
        elif count == 1:
            ax.set(xlim=(xtick[0]-3,xtick[2]+3))
        count += 1
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('weighted_conductance.png', dpi=200)
    plt.clf()

#### Analysis with unweighted network ####

A_u = nx.Graph()
for i in range(len(edges)):
    A_u.add_edge(edges[i,0],edges[i,1])
M_u = nx.to_scipy_sparse_matrix(A_u, nodelist=clusters[:,0], format="csr")
L = csgraph.laplacian(M_u, normed=True)
ivals_u = eigh(L.todense(), eigvals_only=True)
print('Spectrum of unweighted Laplacian: {}\n'.format(ivals_u[1:6]))

conductances_u = np.zeros(int(np.max(clusters[:,1])))
cluster_sizes_u = np.zeros(int(np.max(clusters[:,1])))
for i in range(1,np.max(clusters[:,1])+1):
    S = clusters[np.where(clusters==i)[0],0]
    conductances_u[i-1] = nx.cut_size(A_u, S) / nx.volume(A_u, S)
    cluster_sizes_u[i-1] = S.size
print('Conductances of Louvain clusters (unweighted graph): {}\n'.format(conductances_u))
print('Cluster sizes: {}\n'.format(cluster_sizes_u))
if make_plot:
    plot_data_u = pd.DataFrame(np.vstack([conductances_u, cluster_sizes_u]).T, columns=['Louvain Cond','Louvain Sizes'])
    plot_data_u.sort_values('Louvain Cond', inplace=True, ignore_index=True)

for q in [20, 5, 3]:
    spec_clusters_u = spectral_clustering(q, L, 0) + 1
    spec_conductances_u = np.zeros(int(np.max(spec_clusters_u)))
    spec_cluster_sizes_u = np.zeros(int(np.max(spec_clusters_u)))
    for i in range(1,np.max(spec_clusters_u)+1):
        S = clusters[np.where(spec_clusters_u==i)[0],0]
        spec_conductances_u[i-1] = nx.cut_size(A_u, S) / nx.volume(A_u, S)
        spec_cluster_sizes_u[i-1] = S.size
    print('Conductances of {} spectral clusters (unweighted graph): {}'.format(q,spec_conductances_u))
    print('Cluster sizes: {}\n'.format(spec_cluster_sizes_u))
    if make_plot:
        tmp = pd.DataFrame(np.vstack([spec_conductances_u, spec_cluster_sizes_u]).T, columns=['Spectral Cond','Spectral Sizes'])
        tmp.sort_values('Spectral Cond', inplace=True, ignore_index=True)
        tmp = pd.DataFrame(np.vstack([tmp.values, (-100)*np.ones((20-q,2))]), columns=['Spectral Cond ' + str(q),'Spectral Sizes ' + str(q)])
        plot_data_u = pd.concat([plot_data_u, tmp], axis=1)

# example cluster to explain high conductance
S = np.where(spec_clusters_u==2)[0]
print('Cluster size: {}. Cut Size: {}. Volume: {}'.format(S.size, nx.cut_size(A_u, clusters[S,0]),nx.volume(A_u, clusters[S,0])))
print('Degrees: {}'.format(np.sort([A_u.degree(i) for i in clusters[S,0]])))
print('Superstar\'s out-of-cluster degree: {}'.format(M_u[228,:][:,list(set(range(n))-set(S))].sum()))
print('Number of nodes within cluster who are connected to superstar: {}'.format(M_u[:,228][S,:].sum()))
S_omit = np.array(list(set(S) - set([228])))
print('Omitting the superstar, cut size and volume become {} and {}.'.format(nx.cut_size(A_u, clusters[S_omit,0]),nx.volume(A_u, clusters[S_omit,0])))

if make_plot:
    plot_data_u['Labels'] = range(1,21)

    sns.set_theme(style='whitegrid', font='candara')
    g = sns.PairGrid(plot_data_u, x_vars=plot_data_u.columns[:-1], y_vars=['Labels'], height=6, aspect=.25)
    g.map(sns.stripplot, size=10, orient="h", palette="crest_r", linewidth=1, edgecolor="w")
    titles = ['Louvain', 'Louvain', 'Spectral (L=20)', 'Spectral (L=20)', 'Spectral (L=5)', 'Spectral (L=5)', 'Spectral (L=3)', 'Spectral (L=3)']
    xlabels = ['Conductance', 'Size', 'Conductance', 'Size', 'Conductance', 'Size', 'Conductance', 'Size']
    ylabels = ['Cluster Label', '', '', '', '', '', '', '']
    xticks = [[0-eps,0.5,1], [plot_data_u.iloc[:,1].min(),(plot_data_u.iloc[:,1].max()+plot_data_u.iloc[:,1].min())/2,plot_data_u.iloc[:,1].max()], [0-eps,0.5,1], [plot_data_u.iloc[:,3].min(),(plot_data_u.iloc[:,3].max()+plot_data_u.iloc[:,3].min())/2,plot_data_u.iloc[:,3].max()], [0-eps,0.5,1], [plot_data_u.iloc[0:5,5].min(),(plot_data_u.iloc[:,5].max()+plot_data_u.iloc[0:5,5].min())/2,plot_data_u.iloc[:,5].max()], [0-eps,0.5,1], [plot_data_u.iloc[0:3,7].min(),(plot_data_u.iloc[:,7].max()+plot_data_u.iloc[0:3,7].min())/2,plot_data_u.iloc[:,7].max()]]
    count = 0
    for ax, title, xlabel, ylabel, xtick in zip(g.axes.flat, titles, xlabels, ylabels, xticks):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax.set_xticks(xtick)
        if count % 2 == 0:
            ax.set_xticklabels([0,0.5,1])
            ax.set(xlim=(xtick[0],xtick[2]))
        else:
            ax.set_xticklabels([int(round(i)) for i in xtick])
        if count in [5,7]:
            ax.set(xlim=(xtick[0]-30,xtick[2]+30))
        elif count == 3:
            ax.set(xlim=(xtick[0]-5,xtick[2]+5))
        elif count == 1:
            ax.set(xlim=(xtick[0]-3,xtick[2]+3))
        count += 1
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('unweighted_conductance.png', dpi=200)
    plt.clf()

#### Plot spectra ####

if make_plot:
    ivals = ivals[0:20]
    ivals_u = ivals_u[0:20]
    data = pd.DataFrame( np.hstack([ 
        np.vstack([ivals[1:len(ivals)], range(2,len(ivals)+1)]), 
        np.vstack([ivals_u[1:len(ivals_u)], range(2,len(ivals_u)+1)])
        ]).T, columns=['Eigenvalues','Number'])
    data['Laplacian'] = np.hstack([np.repeat('Weighted', len(ivals)-1), np.repeat('Unweighted', len(ivals_u)-1)])

    sns.set(style='white', font='candara', rc={'figure.figsize':(6.5,4)})
    ax = sns.scatterplot(data=data, x='Number', y='Eigenvalues', hue='Laplacian', s=50)
    ax.set(ylim=(0.05,0.6), xlim=(1,21), xlabel='')
    ax.set_xticks(range(2,len(ivals)+1,2))
    sns.despine(right=True, top=True)
    plt.tight_layout()
    plt.savefig('app_spectra.png', dpi=200)

