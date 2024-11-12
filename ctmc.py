# functions of interest relating to a continuous-time Markov
# chain modeling the stochastic activity of a gene promoter.

import copy
import itertools
import numpy as np
import numdifftools as nd
import utils
from typing import Optional
from ctypes import *
import sys
import os

# matmul_path = os.path.join(sys.path[0], 'cpython/matmul_%s.so' % (sys.platform))
# generate_path = os.path.join(sys.path[0], 'cpython/generate_%s.so' % (sys.platform))
# try:
#     matmul_lib = cdll.LoadLibrary(matmul_path)
#     generate_lib = cdll.LoadLibrary(generate_path)
# except:
#     print('OS %s not recognized' % (sys.platform))

# ctypes_accumulate = matmul_lib.accumulate
# ctypes_generate = generate_lib.generate
# ctypes_accumulate.restype = None
# ctypes_generate.restype = None

##### trajectory generator

class Generator():
    """
    Samples trajectories from a continuous-time Markov chain for
    a fixed number of jumps.
    
    Parameters
    ----------
    Q : np.ndarray
        Rate matrix
    n : int
        Number of jumps per generated trajectory.
    N : int, optional
        Number of simulated trajectories, by default 1.
    r0 : Optional[np.ndarray]
        Initial states, by default None.
    initial_dist : Optional[np.ndarray]
        Initial distribution, by default None. If given, this
        overrides the setting of r0 with a random sample from
        the given distribution.
    seed : int, optional
        Seed for random number generator, by default None.
    """

    def __init__(
        self,
        Q: np.ndarray,
        n: int,
        N: int = 1,
        r0: Optional[np.ndarray] = None,
        initial_dist: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        self.n = n
        self.N = N
        self.initial_dist = initial_dist
        self.rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )
        
        # Main ID of generated trajectories
        self.traj_id = "Continuous-time Markov chain"
        
        if np.ndim(Q) != 2 or np.shape(Q)[0] != np.shape(Q)[1]:
            raise ValueError("Q is expected to be a square matrix")
        else:
            self.Q = Q          # rate matrix
            self.nstates = Q.shape[0]
        
        # Intial conditions
        self.r0 = r0  # initial states
        
        # Initialize variables before simulating
        self._set_simulation_vars()
        self._set_init_cond()
        self._set_noise()
        
    # Simulation parameters and dynamic variables
    def _set_simulation_vars(self):
        self.shape = (self.n, self.N)   # shape of dynamic variables
    
        # Constructing the jump matrix
        self.M = np.zeros_like(self.Q)

        for i in range(self.nstates):
            for j in range(self.nstates):
                if i == j:
                    # i is an absorbing state
                    self.M[i,i] = 1 if self.Q[i,i] == 0 else 0
                else:
                    # either we are stuck in i, or jump to j with p_ij = - q_ij / q_ii
                    self.M[i,j] = 0 if self.Q[i,i] == 0 else - self.Q[i,j] / self.Q[i,i]

        # Dynamic variables
        self.t = np.empty(self.shape, dtype='float')    # hold times
        self.r = np.empty(self.shape, dtype='int')      # states
        self.jumps = np.empty((self.n, self.N, self.nstates), dtype='int')  # jumps
        
    # Set initial conditions
    def _set_init_cond(self):
        if self.initial_dist is not None:
            if np.shape(self.initial_dist) == (self.nstates,):
                
                # Sample initial state from the initial distribution
                self.r0 = self.rng.choice(
                    np.arange(self.nstates), 
                    size=self.N,
                    p=self.initial_dist
                )
                self.r[0] = self.r0
            else:
                raise ValueError(
                    "initial_dist is expected to be an array"
                    f"of shape {(self.nstates,)}"
                )
        elif self.r0 is not None:
            self.r[0] = self.r0      # User input
        else:
            raise ValueError(
                "Either r0 or initial_dist must be given"
            )
    
    # Sample jumps and holding times in advance
    def _set_noise(self):
        self.t = self.rng.exponential(size=self.shape)
                    
        for k in range(self.nstates):
            self.jumps[:, :, k] = self.rng.choice(
                range(self.nstates),
                size=self.shape,
                p=self.M[k]
            )
    
    # Build the CTMC trajectory from the jump and hold times arrays
    def generate(self):
        # index of trajectories
        ind = np.arange(self.N)
        
        for i in range(self.n):
            # vector of self-rates of the current states of all trajectories
            Q_i = self.Q[self.r[i, ind], self.r[i, ind]]
            
            # labels absorbing states
            mask = (Q_i == 0)

            # holding time is infinite at absorbing states
            self.t[i, mask] = np.inf
            
            # holding time is rescaled by the self-rate at all other states
            self.t[i, ~mask] = - self.t[i, ~mask] / Q_i[~mask]

            if i < self.n - 1:
                # next state is selected from the pre-filled jump samples
                self.r[i + 1] = self.jumps[i, ind, self.r[i, ind]]
        
        del(self.jumps)

def generate(rates, c, n, idist, on_ind, nb, na):
    """
    Simulates a CTMC given the rate matrix and initial distribution. If on_ind is given,
    residence times in ON and OFF are returned as well.
    """
    ns=2 ** (nb + na)
    Q=utils.build_mat(rates, c, nb, na).T
    K=-np.diag(Q)
    k=Q+np.diag(K)
    C=np.cumsum(k,0)/K
    flattenC=np.transpose(C).flatten()
    s0=np.random.default_rng().choice(np.arange(ns), p=idist)

    # initialize the ctypes variables
    c_K = (c_double * ns)(*K)
    c_C = (c_double * (ns**2))(*flattenC)
    c_deltat = (c_double * n)()
    c_ton = (c_double * n)()
    c_toff = (c_double * n)()
    c_s = (c_int *n)()
    c_on_ind = (c_int *ns)(*on_ind)

    # simulate
    ctypes_generate(
        c_int(ns), 
        c_int(s0), 
        c_int(n), 
        c_K,
        c_C, 
        c_on_ind,
        c_s, 
        c_deltat,
        c_ton,
        c_toff
    )

    return np.array(c_s), np.array(c_deltat), np.array(c_ton), np.array(c_toff)


##### core functions

def P(Q, t):
    """
    Matrix of transition probabilities at times t of a rate matrix Q,
    where t is an (n, N)-shaped array - n is the number of samples
    of a trajectory and N is the number of trajectories.
    """
    assert np.ndim(Q) == 2
    assert Q.shape[0] == Q.shape[1]
    
    # if t is a float
    if np.ndim(t) == 0:
        N = 1
        n = 1
        t = np.ones((1,1)) * t
    elif np.ndim(t) == 1:
        n = t.shape[0]
        N = 1
        t = t[:, np.newaxis]
    else:
        n, N = t.shape  # nsamples, Ntrajs

    # obtain eigenvalues w and eigenvectors v of Q
    w, v = np.linalg.eig(Q)
    sort_perm = w.argsort()
    w.sort()
    v = v[:, sort_perm]
    m = len(w)  # multiplicity or number of states
    D = np.zeros((n,N,m,m))

    # tensor product of eigenvalues and time array
    wt = np.tensordot(t, w, axes=0)

    # array of diagonal matrices at each timestep and for each trajectory
    D[:, :, np.arange(m), np.arange(m)] = np.exp(wt)
    output = (v @ D) @ np.linalg.inv(v)

    return output.squeeze()

def fpt(Q, hset, t):
    """
    Matrix of PDFs of the first-passage time to hit any state in hset
    """
    nstates = len(Q)
    Q_h = copy.deepcopy(Q)
    Q_h[hset] = np.zeros(nstates)  # rows of hitting set are zeroed

    return Q_h @ P(Q_h, t)

def P_ss(Q):
    """
    Steady-state distribution of a given rate matrix 
    (solution of vQ = 0).
    """
    nstates = len(Q)
    trunc = Q[:,:-1]
    ones = np.ones((nstates, 1))
    A = np.concatenate((trunc, ones), axis=1).T
    b = np.zeros((nstates, 1))
    b[-1] = 1
    
    return np.linalg.solve(A, b).flatten()

def EP_ss(Q):
    """
    Steady-state entropy production rate                              
    """     
    nstates = len(Q)
    states = np.arange(0, nstates, 1)
    
    pi = P_ss(Q)
    EP = 0
                                                                                      
    for i in states:                                                             
        for j in states:
            if Q[j,i] != 0 and pi[j] > 0:                                             
                Jij = pi[i]*Q[i,j] - pi[j]*Q[j,i]                                     
                EP += Jij*np.log(pi[i]*Q[i,j]/(pi[j]*Q[j,i]))                         
                                                    
    return 0.5 * EP

def Z(Q):
    """
    The fundamental matrix of a given rate matrix.
    """ 
    pi = P_ss(Q)
    Pi = np.array([pi for i in pi])

    return np.linalg.inv(Pi - Q) - Pi

def mean_fpt_all(Q):
    """
    The matrix of mean first-passage times between states.
    """
    nstates = len(Q)
    states = np.arange(0, nstates, 1)
    
    Z_ = Z(Q)
    pi = P_ss(Q)
    ET = np.zeros_like(Q)

    for i in states: 
        for j in states:
            ET[j,i] = (Z_[i,i] - Z_[j,i]) / pi[i]

    return ET

def pi_on(weighted_edges, input_edges, c, on_ind):
    """
    Steady-state activation probability at a given concentration
    """
    Q = utils.build_mat(weighted_edges, input_edges, c)
    
    return P_ss(Q) @ on_ind

def var(weighted_edges, input_edges, c, on_ind):
    """
    Asymptotic variance of transcription activity.
    """
    Q = utils.build_mat(weighted_edges, input_edges, c)
    Z_ = Z(Q)
    pi = P_ss(Q)
    
    return 2 * (on_ind * pi) @ (Z_ @ on_ind)

def sharpness(weighted_edges, input_edges, c, on_ind):
    """
    Derivative of the activation function at a given concentration
    """
    f = lambda c : pi_on(weighted_edges, input_edges, c, on_ind)
    
    return c * nd.Derivative(f, order=4)(c)

def ir_acc(weighted_edges, input_edges, c, dc, on_ind):
    """
    Information rate conditioning on accumulated mRNA
    """
    pi_c = pi_on(weighted_edges, input_edges, c, on_ind)
    pi_cdc = pi_on(weighted_edges, input_edges, c + dc, on_ind)
    var_c = var(weighted_edges, input_edges, c, on_ind)
    var_cdc = var(weighted_edges, input_edges, c + dc, on_ind)

    return (pi_cdc - pi_c) ** 2 * (var_cdc + var_c) / (4 * var_c * var_cdc)

def mean_fpt_part(weighted_edges, input_edges, c, on_ind):
    """
    Calculates the mean first passage time from one
    partition to its complement.
    """
    Q = utils.build_mat(weighted_edges, input_edges, c)
    nstates = len(Q)
    off_ind = np.ones(nstates, dtype='int64') - on_ind
    pi = P_ss(Q)
    
    # rate matrix with the ON partition coarse-grained to a single state
    on_cg_rate_mat = utils.coarse_grain(Q, on_ind)
    
    # mean first passage times from any OFF state to ON
    on_mfpt_vec = mean_fpt_all(on_cg_rate_mat)[:-1,-1]
    
    # probability flux going into each OFF state from all ON states
    on_flux_vec = np.compress(off_ind, Q @ (on_ind*pi))
    
    # mean first passage time from OFF to ON
    off_on_mfpt = (on_mfpt_vec @ on_flux_vec) / np.sum(on_flux_vec)
    
    off_cg_rate_mat = utils.coarse_grain(Q, off_ind)
    off_mfpt_vec = mean_fpt_all(off_cg_rate_mat)[:-1,-1]
    off_flux_vec = np.compress(on_ind, Q @ (off_ind*pi))
    on_off_mfpt = (off_mfpt_vec @ off_flux_vec) / np.sum(off_flux_vec)
    
    return on_off_mfpt, off_on_mfpt

def ll_acc(weighted_edges, input_edges, c, on_ind, t_set, t_comp):
    """
    Log-likelihood of an observed duration of transcription activity
    """
    pass

def ll_full(weighted_edges, input_edges, c, ind_set, t_set, t_comp, idist_set):
    """
    Log-likelihood of the full trace of transcription activity
    """
    assert len(t_set) == len(t_comp)

    Q = utils.build_mat(weighted_edges, input_edges, c)

    ind_comp = np.ones_like(ind_set) - ind_set
    _set = np.where(ind_set)[0]
    _comp = np.where(ind_comp)[0]
    
    # matrices of first-passage time PDFs
    R = fpt(Q, _comp, t_set)
    S = fpt(Q, _set, t_comp)
    
    # extract slices of R from on->off and from S from off->on
    R_ = R[..., _set, :][..., _comp]
    S_ = S[..., _comp, :][..., _set]
    
    # concatenate into an alternating on-off sequence of matrices
    D = utils.alternate(R_, S_)
    
    # prepend the initial distribution conditioned on start state
    D.appendleft(idist_set)
    
    # reduce the sequence to a single vector, rescaling along the way
    it = np.array(list(itertools.accumulate(D, matmul_r)))
    
    return np.log(np.sum(it, axis=1))

def ir_full(weighted_edges, input_edges, c, dc, on_ind, idist, n):
    """
    Stochastic estimate of information rate conditioned on the
    full mRNA trajectory
    """
    nstates = len(on_ind)
    off_ind = np.ones(nstates) - on_ind
    on = np.where(on_ind)[0]
    off = np.where(off_ind)[0]

    # building the rate matrix and simulating 
    Q = utils.build_mat(weighted_edges, input_edges, c)
    C = Generator(Q, n, initial_dist=idist)
    C.generate()

    # residence times in on and off partitions
    res_times = utils.get_residence_times(C.r[:,0], C.t[:,0], on)
    t_on = np.array(list(res_times['on'].values()))
    t_off = np.array(list(res_times['off'].values()))

    # only keep nonzero residence times
    start_in_on = t_on[0] > 0
    t_on = t_on[(t_on > 0)]
    t_off = t_off[(t_off > 0)]

    # make res time arrays of equal length
    minlen = min(len(t_on), len(t_off))
    t_on = t_on[:minlen]
    t_off = t_off[:minlen]

    # conditional initial distributions
    idist_on = idist[on] / (idist @ on_ind)
    idist_off = idist[off] / (idist @ off_ind)

    if start_in_on:
        args_ll = (on_ind, t_on, t_off, idist_on)
        time = np.stack((t_on, t_off), axis=1)
    else:
        args_ll = (off_ind, t_off, t_on, idist_off)
        time = np.stack((t_off, t_on), axis=1)

    # cumulative residence time
    time = np.cumsum(np.concatenate(time))

    # log-likelihood ratio
    ll_c = ll_full(weighted_edges, input_edges, c, *args_ll)
    ll_cdc = ll_full(weighted_edges, input_edges, c + dc, *args_ll)
    llr = np.cumsum(ll_c - ll_cdc)

    return t_on, t_off, llr #- llr[1:] / time

def ll_full_fast(rates, c, ind_set, t_set, t_comp, idist_set, nb, na):
    """
    Fast version of ll_full (wraps C function)   !!! OUTDATED !!!
    """
    assert len(t_set) == len(t_comp)

    Q = utils.build_mat(rates, c, nb, na)
    nstates = 2 ** (nb + na)
    ind_comp = np.ones_like(ind_set) - ind_set
    _set = np.where(ind_set)[0]
    _comp = np.where(ind_comp)[0]
    
    # matrices of first-passage time PDFs
    R = fpt(Q, _comp, t_set)
    S = fpt(Q, _set, t_comp)
    
    # extract slices of R from on->off and from S from off->on
    R_ = R[..., _set, :][..., _comp]
    S_ = S[..., _comp, :][..., _set]

    m = len(_set)
    n = len(_comp)
    N_R = R_.shape[0]
    N_S = S_.shape[0]
    N = N_R + N_S

    Rlist = R_.flatten()
    Slist = S_.flatten()

    c_m = c_int(m)
    c_n = c_int(n)
    c_N = c_int(N)
    c_v = (c_double * m)(*idist_set)
    c_Rlist = (c_double * (m*n*N_R))(*Rlist)
    c_Slist = (c_double * (m*n*N_S))(*Slist)
    c_vlist = (c_double * (m*N))()

    # compute the log-likelihood using the forward equation
    ctypes_accumulate(c_m, c_n, c_N, c_v, c_Rlist, c_Slist, c_vlist)

    # reshape c_vlist into (njumps, 2)-shaped array
    llr_vec = np.reshape(c_vlist, (N, 2))

    return np.log(np.sum(llr_vec, axis=1))
   
def ir_full_fast(rates, c, dc, on_ind, idist, n, nb, na):
    """
    Fast version of ir_full (partially implemented in C)   !!! OUTDATED !!!
    """
    nstates = 2 ** (nb + na)
    off_ind = np.ones(nstates) - on_ind
    on = np.where(on_ind)[0]
    off = np.where(off_ind)[0]

    # building the rate matrix and simulating 
    Q = utils.build_mat(rates, c, nb, na)
    _, _, t_on, t_off = generate(rates, c, n, idist, on_ind, nb, na)

    # # only keep nonzero residence times
    start_in_on = t_on[0] > 0

    # total duration of trajectory
    t = np.sum(t_on + t_off)

    # conditional initial distributions
    idist_on = idist[on] / (idist @ on_ind)
    idist_off = idist[off] / (idist @ off_ind)

    if start_in_on:
        args_ll = (on_ind, t_on, t_off, idist_on, nb, na)
    else:
        args_ll = (off_ind, t_off, t_on, idist_off, nb, na)

    # log-likelihood ratio
    ll_c = ll_full_fast(rates, c, *args_ll)
    ll_cdc = ll_full_fast(rates, c + dc, *args_ll)
    llr = np.cumsum(ll_c - ll_cdc)

    return t_on, t_off, - llr / t


###### convenience functions

def matmul_r(v, M):
    """
    Matrix multiplication rescaling the right argument
    """
    return np.matmul(v / np.sum(v), M)

def ep_rate_array(arr, c, nb, na, nrates, recover_rates=None):
    """
    Calculates the EP rate for each circuit on an optimization
    run returned by CMA-ES
    """
    nrow, _ = arr.shape
    ep_arr_opt = np.zeros(nrow)

    # calculate EP for these circuits
    for i in range(nrow):
        rates = 10.0 ** arr[i,5:5+nrates]

        if recover_rates is not None:
            rates = recover_rates(rates, nb, na)

        Q = utils.build_mat(rates, c, nb, na)
        ep_arr_opt[i] = EP_ss(Q)

    return ep_arr_opt