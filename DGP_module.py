import numpy as np, networkx as nx, pandas as pd, math
from scipy import spatial
from scipy.sparse.linalg import inv
from scipy.special import gamma as GammaF
from scipy.sparse import csr_matrix, identity
from scipy.stats import hypergeom

def ball_vol(d,r):
    """Computes the volume of a d-dimensional ball of radius r. Used to construct RGG.

    Parameters
    ----------
    d : int 
        Dimension of space.
    r : float
        RGG parameter.
    """
    return math.pi**(d/2) * r**d / GammaF(d/2+1)

def gen_RGG(positions, r):
    """Generates an RGG

    Parameters
    ----------
    positions : numpy array
        n x d array of d-dimensional positions, one for each of the n nodes.
    r : float
        RGG parameter.

    Returns
    -------
    RGG as NetworkX graph
    """
    kdtree = spatial.cKDTree(positions)
    pairs = kdtree.query_pairs(r) # Euclidean norm
    RGG = nx.empty_graph(n=positions.shape[0], create_using=nx.Graph())
    RGG.add_edges_from(list(pairs))
    return RGG

def gen_Y(A_mat, errors):
    """Generates outcomes for first simulation study.

    Parameters
    ----------
    A_mat : scipy sparse matrix
        n x n adjacency matrix
    errors : numpy array
        n-dimensional array of error terms

    Returns
    -------
    n-dimensional array of outcomes
    """
    n = errors.size
    degrees = A_mat.dot(np.ones(n))
    r,c = A_mat.nonzero()
    rD_sp = csr_matrix(((1.0/np.maximum(degrees,1))[r], (r,c)), shape=(A_mat.shape))
    A_norm = A_mat.multiply(rD_sp) # row-normalized adjacency matrix
    errbar = A_norm.dot(errors)
    Y = errors + errbar
    return Y, A_norm

def linear_in_means(D, A_norm, errors, theta):
    """Generates outcomes from linear-in-means model.

    Parameters
    ----------
    D : numpy array
        n-dimensional vector of treatment indicators.
    A_norm : scipy sparse matrix (csr format)
        Row-normalized adjacency matrix.
    errors : numpy array
        n-dimensional array of error terms
    theta : numpy array
        Vector of structural parameters: intercept, endogenous peer effect, exogenous peer effect, treatment effect.

    Returns
    -------
    n-dimensional array of outcomes
    """
    LIM_inv = inv( identity(D.size,format='csc') - theta[1]*A_norm ) 
    Y = LIM_inv.dot( (theta[0] + theta[2]*A_norm.dot(D) + theta[3]*D + errors) )
    return Y

def threshold_model(D, A_norm, errors, theta):
    """Generates outcomes according to threshold model of social influence with strategic complements.

    Parameters
    ----------
    D : numpy array
        n-dimensional vector of treatment indicators.
    A_norm : scipy sparse matrix (csr format)
        Row-normalized adjacency matrix.
    errors : numpy array
        n-dimensional array of error terms
    theta : numpy array
        Vector of structural parameters: intercept, endogenous peer effect, exogenous peer effect, treatment effect.

    Returns
    -------
    n-dimensional array of outcomes
    """
    if theta[1] < 0:
        raise ValueError('Must have theta[1] >= 0.')

    U_exo_eps = theta[0] + theta[2]*A_norm.dot(D) + theta[3]*D + errors

    # set initial outcome to 1 iff the agent will always choose outcome 1
    Y = (U_exo_eps > 0).astype('float')

    stable = False
    while stable == False:
        peer_avg = np.squeeze(np.asarray(A_norm.dot(Y[:,None])))
        Y_new = (U_exo_eps + theta[1]*peer_avg > 0).astype('float') # best response
        if (Y_new == Y).sum() == D.size:
            stable = True
        else:
            Y = Y_new

    return Y_new

def assemble_data():
    """Clean and assemble data for empirical application.

    Returns
    -------
    Y : numpy array
        n-dimensional vector of outcomes.
    D : numpy array
        n-dimensional vector of treatment indicators.
    A : NetworkX graph
        Network.
    A_norm : scipy sparse matrix
        n x n row-normalized adjacency matrix.
    pscores0 : numpy array
        n-dimensional vector of propensity scores for probability of not being treated conditional on network.
    IDs : numpy array
        Student and school identifiers for each observation.
    """
    missing_values = ['--blank--','--impossible--','--shifted--','--nom--','--void--','-55','-66','-77','-88','-95','-96','-97','-98','-99','999',' ']
    cols = ['UID','ID','SCHID','STRB','WRISTOW2','TREAT','SCHTREAT']+['ST'+str(i) for i in range(1,11)]
    data = pd.read_csv('37070-0001-Data.tsv', sep='\t', usecols=cols, na_values = missing_values)
    data.fillna(-99, inplace=True)
    data = data[data.SCHTREAT == 1] # restrict to treated schools only
    data = data[data.STRB >= 0]
    data = data[data.ID >= 0]
    data = data[data.SCHID >= 0]
    data = data[data.WRISTOW2 >= 0]
    data.at[271,'ID'] = 284 # fix typo

    data.sort_values('SCHID',inplace=True)
    data = data[(data.SCHID == 24) | (data.SCHID == 22) | (data.SCHID == 60) | (data.SCHID == 56) | (data.SCHID == 58)] # restrict sample to these schools

    # construct graph
    adjlist = np.hstack([data[['ID','SCHID']+['ST'+str(i) for i in range(1,11)]].values]).astype('int') 
    school_ids = np.unique(adjlist[:,1]) 
    A = nx.DiGraph()
    for i in data.UID:
        A.add_node(i)
    for sch in school_ids:
        # add edges separately for each school
        school_adjlist = adjlist[adjlist[:,1] == sch,:]
        for i in range(school_adjlist.shape[0]):
            A.add_node(sch*100000+school_adjlist[i,0]) # incorporate school ID into unit ID to get UID
            for col in range(2,12):
                if school_adjlist[i,col] in school_adjlist[:,0]:
                    A.add_edge( sch*100000+school_adjlist[i,0], sch*100000+school_adjlist[i,col] )

    Y = data.WRISTOW2.values
    D = data.TREAT.values
    D[D==0] = (-1)*np.ones(Y.size)[D==0]  # recode ineligible to -1 
    D[D==2] = np.zeros(Y.size)[D==2]   # recode control to 0

    # construct normalized adjacency matrix
    A_mat = nx.to_scipy_sparse_matrix(A, nodelist=np.squeeze(data[['UID']].values).tolist())
    out_degrees = np.squeeze(A_mat.dot(np.ones(A_mat.shape[0])[:,None]))
    r,c = A_mat.nonzero()
    rD_sp = csr_matrix(((1.0/np.maximum(out_degrees,1))[r], (r,c)), shape=(A_mat.shape))
    A_norm = A_mat.multiply(rD_sp)

    # construct propensity score for having zero treated friends
    num_friends_blks = np.vstack([np.squeeze(np.asarray( A_mat.dot((data.STRB==k).to_numpy()[:,None]) )) for k in range(1,5)]) # each row is a vector giving the number of friends of each student assigned to student block k
    pscores0 = hypergeom(16, 8, num_friends_blks).pmf(0).prod(axis=0) # blocks of 16 students, half treated

    # relabel nodes
    IDs = data[['UID','SCHID']].values
    mapping = dict(zip(A,range(IDs.shape[0])))
    A = nx.relabel_nodes(A, mapping)

    return Y,D,A,A_norm,pscores0,IDs

