# helper functions for building and manipulating CTMC rate 
# matrices, creating experiments and handling simulation data

import os
import json
import numpy as np
from collections import deque
import networkx as nx

############## rate matrix utilities

def build_mat(weighted_edges, input_edges, c):
    """
    Builds a rate matrix from a weighed list of edges where
    the edges specified in `input_edges` are concentration-dependent

    weighted_edges : weighed edge list
    input_edges : edges that depend on concentration
    c : concentration
    """
    nstates = int(np.max(weighted_edges[:,:-1])) + 1

    # create adjacency matrix
    G = nx.DiGraph()
    G.add_nodes_from(range(nstates))
    G.add_weighted_edges_from(weighted_edges)
    adj_mat = nx.to_numpy_array(G)

    # update c-dependent entries
    for u, v, in input_edges:
        adj_mat[u, v] *= c

    # add self-rates to make rows sum to zero
    diag = np.sum(adj_mat, 1)

    return adj_mat - np.diag(diag)

def coarse_grain(Q, in_ind):
    """
    Returns a matrix with the complement set of the indicated
    states coarse-grained into a single state.
    """
    nstates = len(Q)
    
    # indicator array for the complement of the selected partition
    out_ind = np.ones(nstates, dtype='int64') - in_ind
    n_in = np.sum(in_ind)
    M = np.zeros((n_in+1, n_in+1))
    
    # extract rates within partition
    M[:-1, :-1] = np.compress(in_ind, np.compress(in_ind, Q, axis=0), axis=1)

    # last row is sum of incoming rates
    M[-1, :-1] = np.sum(np.compress(in_ind, np.compress(out_ind, Q, axis=0), axis=1), axis=0)

    # last col is sum of outgoing rates
    M[:-1, -1] = np.sum(np.compress(out_ind, np.compress(in_ind, Q, axis=0), axis=1), axis=1)

    # diagonal is such that rows sum to zero
    M_diag_zero = M - np.diag(np.diag(M))
    M_rowsum = np.sum(M_diag_zero, axis=1)
    
    return M_diag_zero - np.diag(M_rowsum)

def get_residence_times(x, t, on_states):
    """
    Returns the residence times of a trajectory in each of two                            
    subsets, which need not cover the whole state space.                                                                                                                 
    """                                                                                   
    residence_times = {'on': {0: 0}, 'off': {0: 0}}                                       
    jump_index = 0 
    previous_state = x[0]                                                                 
    
    for state, time in zip(x, t):                                                         
        if state in on_states:
            if previous_state not in on_states:                                           
                jump_index += 1
                residence_times['on'][jump_index] = 0                                     
                previous_state = state
            residence_times['on'][jump_index] += time                                     
        
        else:
            if previous_state in on_states:
                residence_times['off'][jump_index] = 0                                    
                previous_state = state
            residence_times['off'][jump_index] += time                                    
        
        # set final off residence time to zero to keep index range equal                  
        if jump_index not in residence_times['off'].keys():                               
            residence_times['off'][jump_index] = 0                                        
    
    return residence_times

def alternate(a, b):
    """
    Returns an alternating sequence of elements from a and b
    """
    if len(a) != len(b):
        raise ValueError("a and b must be of equal length")

    out = deque()
    n = len(a)

    for k in range(n):
        out.append(a[k])
        out.append(b[k])

    return out

############## experiment utilities

def create_experiment(params, name):
    """
    Dumps the parameters specified in params in a
    json file in exps/spec and creates a directory
    in exps/data with the experiment name

    Parameters:
    -----------
    params : dict
        Experiment parameters
    name : str
        Experiment name
    """

    datapath = os.path.join('exps/data', name)
    specpath = os.path.join('exps/spec', name)

    with open(specpath+'.json', 'w') as f:
        json.dump(params, f)

    if not os.path.exists(datapath):
        os.makedirs(datapath)

def cmaes_output_to_arr(file):
    """
    Reads data from outcmaes/xrecentbest.dat into an array
    """
    with open(file, 'r') as f:
        s = f.read()

    s = s.split("\n")
    l = s[1].split(' ')
    ncol = len(l)
    nrow = len(s)
    arr = np.zeros((nrow-2, ncol))

    for key, val in enumerate(s):
        if key != 0 and len(val) != 0:
            arr[key-1] = np.array([float(x) for x in val.split(' ')])

    return arr

# change this so it can read output of both pareto.opt.dev and specificity.opt.dev

def get_data(dirname, whichf, whicha="pareto", skipkeys=[]):
    paramfile = "params"
    path = '{}/.julia/dev/LLROpt/data/'.format(os.environ["HOME"])
    params = json.load(open(path+'{}/{}.json'.format(dirname, paramfile)))

    data = {}
    data_proc = {}
    ncpus = params["ncpus"]
    nbetas = params["nparams"]
    nparams_per_cpu = int(nbetas/ncpus)

    if whicha == "pareto":
        whichalg = "_vs_ep_pareto_pid"
    elif whicha == "spec":
        whichalg = "_mix_vs_pure_pareto_pid"
    else:
        raise ValueError('argument whicha should be either pareto or spec' )

    # merge data from each process into a single dict
    for cpukey in range(1, ncpus + 1):
        if cpukey not in skipkeys:
            data_proc = json.load(open(path+'{}/ir_{}{}{}.json'.format(dirname, whichf, whichalg, cpukey)))

            for i in range(1, nparams_per_cpu+1):
                keyb = (cpukey - 1) * nparams_per_cpu + i
                data[keyb] = data_proc[str(keyb)]

    return data, params

def find_halfmax(f, target=0.5, lb=-5, ub=5, tol=1e-4, verbose=False):
    """
    Binary search for a given value of f, assuming it's monotonically increasing.
    """
    c_k = (ub + lb) / 2
    f_k = f(c_k)
    
    while np.abs(f_k - target) > tol:
        
        if f_k > target:
            ub = c_k
        elif f_k <= target:
            lb = c_k
        else:
            raise ValueError("something's wrong with f")
            
        c_k = (ub + lb) / 2
        f_k = f(c_k)
        
        if verbose:
            print(c_k, f_k)
        
    return c_k, f_k