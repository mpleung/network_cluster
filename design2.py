import numpy as np, pandas as pd, networkx as nx, multiprocessing as mp, sys, traceback, os
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix, identity, csgraph
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm, binom
from DGP_module import *
from inference_module import *

if not os.path.isdir('results'): os.mkdir('results')

##### User parameters #####

processes = 32                           # number of cores to use
B = 5000                                 # number of simulations
sigf_lvl = 0.05                          # significance level of t-test
nclus = 8
num_schools = [1,2,4]                    # number of schools to include in sample
p = 0.5                                  # treatment probability
theta_LIM = np.array([-1,0.8,1,1])       # structural parameters for linear-in-means: intercept, 
                                         #   endogenous, exogenous, and treatment effect
theta_TSI = np.array([-1,1.5,1,1])       # structural parameters for threshold model: intercept, 
                                         #   endogenous, exogenous, and treatment effect
network_model = 'RGG'                    # options: config, RGG, RCM, config
save_csv = True                          # save output in CSV
estimands_only = False                   # only simulate estimands
manual_estimands = True                  # use previously simulated estimands (hard-coded below)

print('theta: {},{}'.format(theta_LIM,theta_TSI))

##### Task per node #####

def one_sim(b, deg_seq, eligibles, estimates_only, estimand_LIM, estimand_TSI):
    """
    Task to be parallelized: one simulation draw. Set estimates_only to True if you only want to return estimators.
    """
    n = deg_seq.size
    c = 2 if estimates_only else 1
    seed = int(n*c*(B*10) + b)
    np.random.seed(seed=seed)
    if b%100 == 0: 
        print('  b = {}'.format(b))
        sys.stdout.flush()

    # simulate network and errors
    if network_model == 'config':
        A = nx.configuration_model(deg_seq, seed=seed)
        A = nx.Graph(A) # remove multi-edges
        A.remove_edges_from(nx.selfloop_edges(A)) # remove self-loops
        errors = np.random.normal(size=n)
    elif network_model == 'RGG':
        positions = np.random.uniform(size=(n,2))
        A = gen_RGG(positions, (deg_seq.mean()/ball_vol(2,1)/n)**(1/2))
        errors = np.random.normal(size=n) + (positions[:,0] - 0.5)
    elif network_model == 'RCM':
        positions = np.random.uniform(size=(n,2))
        alpha = np.random.uniform(size=n)
        r_n = (deg_seq.mean()/3.5/ball_vol(2,1)/n)**(1/2)
        latent_index = alpha + alpha[:,None] - squareform(pdist(positions / r_n))
        P_LSM = np.exp(latent_index) / (1 + np.exp(latent_index)) # n x n matrix of link probabilities
        np.fill_diagonal(P_LSM,0) # zero diagonals
        U = np.random.uniform(0,1,size=(n,n))
        U = np.tril(U) + np.tril(U, -1).T
        A = nx.from_numpy_matrix((U < P_LSM).astype(int))
        errors = np.random.normal(size=n) + (positions[:,0] - 0.5)
    else:
        raise ValueError('Not a valid choice of network model.')
    A_mat = nx.to_scipy_sparse_matrix(A, nodelist=range(n), format='csc')
    deg_seq_sim = np.squeeze(A_mat.dot(np.ones(n)[:,None]))
    r,c = A_mat.nonzero() 
    rD_sp = csr_matrix(((1.0/np.maximum(deg_seq_sim,1))[r], (r,c)), shape=(A_mat.shape))
    A_norm = A_mat.multiply(rD_sp) # row-normalized adjacency matrix
    friends_eligible = np.squeeze(np.asarray(A_mat.dot(eligibles[:,None])))

    # simulate treatments and outcomes
    D = np.zeros(n)
    D[eligibles] = np.random.binomial(1,p,eligibles.sum()) # assign treatments to eligibles
    Y_LIM = linear_in_means(D, A_norm, errors, theta_LIM)
    Y_TSI = threshold_model(D, A_norm, errors, theta_TSI)
    friends_treated = np.squeeze(np.asarray(A_mat.dot(D[:,None]))) # num friends treated

    # compute pscores and Zs for estimator
    pop = (friends_eligible > 0) # indicators for inclusion in population, in this case only include units with eligible friends
    real_N = pop.sum()
    pscores0 = binom(friends_eligible,p).pmf(0)
    pscores1 = 1 - binom(friends_eligible,p).pmf(0)
    ind1 = friends_treated > 0 # exposure mapping indicators for spillover effect
    ind0 = 1 - ind1
    chin_est_LIM, chin_est_TSI = 0, 0
    chin_est_TSI = Y_TSI[pop][ind1[pop]==1].mean() - Y_TSI[pop][ind0[pop]==1].mean()
    Zs_LIM = make_Zs(Y_LIM,ind1,ind0,pscores1,pscores0,pop)
    Zs_TSI = make_Zs(Y_TSI,ind1,ind0,pscores1,pscores0,pop)

    if estimates_only:
        # estimators
        estimate_LIM = Zs_LIM[pop].mean()
        estimate_TSI = Zs_TSI[pop].mean()
        return [estimate_LIM, estimate_TSI]
    else:
        # construct clusters, measure conductance
        A_components = [A.subgraph(c).copy() for c in sorted(nx.connected_components(A), key=len, reverse=True)]
        A_giant = A_components[0] # extract giant component
        L = csgraph.laplacian(nx.to_scipy_sparse_matrix(A_giant, format="csr"), normed=True)
        clusters = spectral_clustering(nclus, L, seed) + 1
        phi, clusters = conductance(clusters, A, A_giant)
        for C in A_components[1:len(A_components)]:
            if C.number_of_nodes() >= 20:
                clusters[list(C)] = np.max(clusters) + 1 # assign cluster labels to nontrivial components
            else:
                clusters[list(C)] = -1 # trivial components have label -1
        actual_num_clusters = np.unique(clusters).size - 1
        cluster_sizes = np.sort(np.histogram(clusters, np.max(clusters))[0])
        cluster_size_1 = cluster_sizes[cluster_sizes.size-1]
        cluster_size_2 = cluster_sizes[cluster_sizes.size-2]
        cluster_size_l = cluster_sizes[cluster_sizes >= 20][0]

        # randomization test
        thetahat_LIM = sample_means(Zs_LIM[pop], clusters[pop])
        rtest_LIM = rand_test(thetahat_LIM, estimand_LIM, sigf_lvl)
        thetahat_TSI = sample_means(Zs_TSI[pop], clusters[pop])
        rtest_TSI = rand_test(thetahat_TSI, estimand_TSI, sigf_lvl)

        # HAC t-test
        thetahat_LIM_full = sample_means(Zs_LIM[pop], np.ones(pop.sum()))
        thetahat_TSI_full = sample_means(Zs_TSI[pop], np.ones(pop.sum()))
        HAC_SE_LIM = np.sqrt(network_HAC(Zs_LIM, A, 1, pop) / pop.sum())
        HAC_SE_TSI = np.sqrt(network_HAC(Zs_TSI, A, 1, pop) / pop.sum()) 
        ttest_LIM = np.abs(thetahat_LIM_full - estimand_LIM) / HAC_SE_LIM > norm.ppf(1-sigf_lvl/2)
        ttest_TSI = np.abs(thetahat_TSI_full - estimand_TSI) / HAC_SE_TSI > norm.ppf(1-sigf_lvl/2)

        return [thetahat_LIM.mean(), thetahat_TSI.mean(), rtest_LIM, rtest_TSI, ttest_LIM, ttest_TSI, phi, actual_num_clusters, cluster_size_1, cluster_size_2, cluster_size_l, real_N]

##### Storage #####

thetahat_LIM = np.zeros(len(num_schools))
thetahat_TSI = np.zeros(len(num_schools))
rtest_LIM = np.zeros(len(num_schools))
rtest_TSI = np.zeros(len(num_schools))
ttest_LIM = np.zeros(len(num_schools))
ttest_TSI = np.zeros(len(num_schools))
phi = np.zeros(len(num_schools))
actual_num_clusters = np.zeros(len(num_schools))
cluster_size_1 = np.zeros(len(num_schools))
cluster_size_2 = np.zeros(len(num_schools))
cluster_size_l = np.zeros(len(num_schools))
Ns = np.zeros(len(num_schools)).astype('int')

##### Main #####

# assemble network data
_,D,A,_,_,IDs = assemble_data()
deg_seq = np.array([i[1] for i in A.out_degree])
A = A.to_undirected()
eligibles = (D >= 0)

for i,ns in enumerate(num_schools):
    # select schools
    if ns == 1:
        students = (IDs[:,1] == 24)
    elif ns == 2:
        students = (IDs[:,1] == 24) + (IDs[:,1] == 22)
    else:
        students = (IDs[:,1] == 24) + (IDs[:,1] == 22) + (IDs[:,1] == 60) + (IDs[:,1] == 56)
    print('n = {}'.format(students.sum()))

    if deg_seq[students].sum() % 2 != 0: 
        deg_seq_pop = deg_seq[students].copy()
        deg_seq_pop[0] += 1 # need even total degree for configuration model
    else:
        deg_seq_pop = deg_seq[students]

    if manual_estimands:
        # HARD CODED simulated estimands and oracle SEs from runs with estimands_only=True
        if ns == 1:
            if network_model == 'config':
                estimands = np.array([0.32880273382045605,0.0760735439769643]) 
            elif network_model == 'RGG':
                estimands = np.array([0.6955800715987678,0.09584027762253172]) 
            elif network_model == 'RCM':
                estimands = np.array([0.44055909736938265,0.08181188545067478]) 
        elif ns == 2:
            if network_model == 'config':
                estimands = np.array([0.34141409230292163,0.07500066547398646]) 
            elif network_model == 'RGG':
                estimands = np.array([0.6808656267764969,0.09713639239291233]) 
            elif network_model == 'RCM':
                estimands = np.array([0.43312436260328213,0.07981135809094332]) 
        elif ns == 4:
            if network_model == 'config':
                estimands = np.array([0.31211856164758434,0.07963784112687888]) 
            elif network_model == 'RGG':
                estimands = np.array([0.6984956530348513,0.10020642097507103]) 
            elif network_model == 'RCM':
                estimands = np.array([0.42401452551362573,0.08073738521554417]) 
        else:
            estimands = np.array([0,0])
    else:
        # simulate only estimands and oracle standard errors
        def one_sim_wrapper(b):
            try:
                return one_sim(b, deg_seq_pop, eligibles[students], True, 0, 0)
            except:
                print('%s: %s' % (b, traceback.format_exc()))
                sys.stdout.flush()
        
        sims_range = range(B,2*B)
        pool = mp.Pool(processes=processes, maxtasksperchild=1)
        parallel_output = pool.imap(one_sim_wrapper, sims_range, chunksize=25) 
        pool.close()
        pool.join()
        results = np.array([r for r in parallel_output])
        estimands = results.mean(axis=0)

    print('Estimands: LIM={}, TSI={}'.format(estimands[0],estimands[1])) # use these to HARD CODE
    sys.stdout.flush()

    if estimands_only:
        results = np.zeros(12)
    else:
        # simulate main results
        def one_sim_wrapper(b):
            try:
                return one_sim(b, deg_seq_pop, eligibles[students], False, estimands[0], estimands[1])
            except:
                print('%s: %s' % (b, traceback.format_exc()))
                sys.stdout.flush()
        
        sims_range = range(B)
        pool = mp.Pool(processes=processes, maxtasksperchild=1)
        parallel_output = pool.imap(one_sim_wrapper, sims_range, chunksize=25)
        pool.close()
        pool.join()
        results = np.array([r for r in parallel_output])
        results = results.mean(axis=0)

    # store results
    thetahat_LIM[i] = results[0]
    thetahat_TSI[i] = results[1]
    rtest_LIM[i] = results[2]
    rtest_TSI[i] = results[3]
    ttest_LIM[i] = results[4]
    ttest_TSI[i] = results[5]
    phi[i] = results[6]
    actual_num_clusters[i] = results[7]
    cluster_size_1[i] = results[8]
    cluster_size_2[i] = results[9]
    cluster_size_l[i] = results[10]
    Ns[i] = results[11]

##### Output #####

print('Estimates: LIM={}, TSI={}'.format(thetahat_LIM,thetahat_TSI))

# print table
table = pd.DataFrame(np.vstack([ rtest_LIM, ttest_LIM, rtest_TSI, ttest_TSI, phi, actual_num_clusters, cluster_size_1, cluster_size_2, cluster_size_l, Ns ]))
table.index = ['LIM Rand', 'LIM HAC', 'TSI Rand', 'TSI HAC', '$\max_\ell \phi(\C_\ell)$', '\# Clusters', '1st Clus.', '2nd Clus.', 'Last Clus.', '$n$']
table.columns = num_schools
print(table.to_latex(float_format = lambda x: '%.4f' % x, header=True, escape=False))

# save table
if save_csv:
    table.to_csv('results/results_test_SE_' + network_model + '.csv', float_format='%.6f')

