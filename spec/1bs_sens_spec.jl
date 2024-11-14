### Biophysical parameters
a = 3e-3        # um
D_tf = 7.4      # um^2/s
D_cplx = 0.1    # um^2/s
cr = 5.6        # um^-3
dc = 0.1 * cr
dl_tf = 4*a*D_tf
ub = 5
lb = 1e-5
max_ir_full = 0.5 * (dc/cr)^2 * (cr*dl_tf*ub)/(cr*dl_tf + ub)

### Upper bounds for pure ligand circuit
llb1 = ones(8) .* log10(lb)
lub1 = ones(8) .* log10(ub)
lub1[1] = log10(dl_tf)
lub1[3] = log10(dl_tf)

### Upper bounds for mixed ligands circuit
llb2 = ones(14) .* log10(lb)
lub2 = ones(14) .* log10(ub)
lub2[2] = log10(dl_tf)
lub2[3] = log10(dl_tf)
lub2[6] = log10(dl_tf)
lub2[7] = log10(dl_tf)

### Circuit descriptions
edges1 = [(1,3), (1,2), (2,4), (2,1), (3,1), (3,4), (4,2), (4,3)]
edges2 = [(5,1), (1,5), (2,6), (6,2), (1,2), (1,3), (2,4),
          (2,1), (3,1), (3,4), (4,2), (4,3), (5,6), (6,5)]
input_edges = [(1,3), (2,4)]
on_ind1 = BitArray([0,1,0,1])
on_ind2 = BitArray([0,1,0,1,0,1])
idist1 = ones(4) ./ 4
idist2 = ones(6) ./ 6

### Optimization parameters
maxiters_acc = 40000
maxiters_full = 3000
xinit1 = (lub1 .+ llb1) ./ 2
xinit2 = (lub2 .+ llb2) ./ 2
we1 = [(u,v,r) for ((u,v),r) in zip(edges1, xinit1)]
we2 = [(u,v,r) for ((u,v),r) in zip(edges2, xinit2)]
n = 1000000
N = 2000
k = 1
whichf = "acc"

### Temperature parameter
betas = map(x->tan(x), range(0.0, stop=3.141592/2000, length=512))
nparams = length(betas)

### Other parameters
# gammas = [1.5, 2.5, 5, 10, 50, 500, 5000, 50000]
gammas = [1.5, 2.5, 10]
# gammas = range(1, stop=4, length=10)
# wvals = [1e-5, 2, 5, 10, 50, 500]
wvals = [2.5, 500]

### objective functions

# returns - IR_acc_mix - beta * IR_acc_pure - gamma * EP
function target_acc(logrates, args...)
    edges1, edges2, iedges, cr, dc, on_ind1, on_ind2, beta, gamma, w = args
    logcw = log10(cr) + log10(w)
    kboff, konu, kbon, koffu, kuoff, konb, kuon, koffb = logrates
    logrates2 = [kuoff+log10(gamma), kboff+logcw, kbon+logcw, kuon+log10(gamma), 
        konu, kboff, kbon, koffu, kuoff, konb, kuon, koffb, konb, koffb]

    wedges1 = [(u,v,10 ^ r) for ((u,v),r) in zip(edges1, logrates)]
    wedges2 = [(u,v,10 ^ r) for ((u,v),r) in zip(edges2, logrates2)]

    Q1 = build_mat(wedges1, iedges, cr)
    Q2 = build_mat(wedges2, iedges, cr)
    # EP = EP_ss(Q1)
    EP = EP_ss(Q2)

    Jplus = 0
    pi = P_ss(Q1)
    pi2 = P_ss(Q2)
    J = pi[1] * Q1[1,2] - pi[2] * Q1[2,1]
    Jplus = J + abs(J)   # penalty for counterclockwise flux
    ir_acc_pure = ir_acc(wedges1, iedges, cr, dc, on_ind1)
    ir_acc_mix = ir_acc(wedges2, iedges, cr, dc, on_ind2)
    mlog_epsilon = log2((pi2[4])/(pi2[6]))

    # (- ir_acc_pure - beta*ir_acc_mix + Jplus, ir_acc_mix, ir_acc_pure, EP)
    (- ir_acc_mix - beta*mlog_epsilon + Jplus, mlog_epsilon, ir_acc_mix, EP)
    
end

# returns - IR_full_mix - beta * IR_full_pure - gamma * EP
function target_full(logrates, args...)
    m1, m2, beta, gamma, w = args
    logcw = log10(m1.c) + log10(w)
    kboff, konu, kbon, koffu, kuoff, konb, kuon, koffb = logrates
    logrates2 = [kuoff+log10(gamma), kboff+logcw, kbon+logcw, kuon+log10(gamma), 
        konu, kboff, kbon, koffu, kuoff, konb, kuon, koffb, konb, koffb]

    for (keyb, val) in enumerate(logrates)
        (u,v,r) = m1.weighted_edges[keyb]
        m1.weighted_edges[keyb] = (u,v,10 ^ val)
    end

    for (keyb, val) in enumerate(logrates2)
        (u,v,r) = m2.weighted_edges[keyb]
        m2.weighted_edges[keyb] = (u,v,10 ^ val)
    end

    init!(m1)
    init!(m2)
    EP = EP_ss(m1.Q_c)

    Jplus = 0
    pi = P_ss(m1.Q_c)
    J = pi[1] * m1.Q_c[1,2] - pi[2] * m1.Q_c[2,1]
    Jplus = J + abs(J)   # penalty for counterclockwise flux
    # ir_full!(m1)
    ir_full!(m2)
    nskip = 5

    # empirical mean estimator
    # ir_full_pure = m1.llr[end] / sum(m1.rt)
    ir_full_mix = m2.llr[end] / sum(m2.rt)

    (- ir_full_pure - beta*ir_full_mix + Jplus, ir_full_mix, ir_full_pure, EP)
end
