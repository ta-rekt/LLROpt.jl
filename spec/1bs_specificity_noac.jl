### Biophysical parameters
a = 3e-3        # um
D_tf = 7.4      # um^2/s
D_cplx = 0.1    # um^2/s
cr = 5.6        # um^-3
dc = 0.1 * cr
dl_tf = 4*a*D_tf
ub = 5
lb = 1e-5

### Upper bounds for pure ligand circuit
llb1 = ones(2) .* log10(lb)
lub1 = ones(2) .* log10(ub)
lub1[1] = log10(dl_tf)

### Upper bounds for mixed ligands circuit
llb2 = ones(4) .* log10(lb)
lub2 = ones(4) .* log10(ub)
lub2[1] = log10(dl_tf)
lub2[3] = log10(dl_tf)

### Circuit descriptions
edges1 = [(1,2), (2,1)]
edges2 = [(1,2), (2,1), (1,3), (3,1)]
input_edges = [(1,2)]
on_ind1 = BitArray([0,1])
on_ind2 = BitArray([0,1,1])
idist1 = ones(2) ./ 2
idist2 = ones(3) ./ 3

### Optimization parameters
maxiters_acc = 20000
maxiters_full = 3000
xinit1 = (lub1 .+ llb1) ./ 2
xinit2 = (lub2 .+ llb2) ./ 2
we1 = [(u,v,r) for ((u,v),r) in zip(edges1, xinit1)]
we2 = [(u,v,r) for ((u,v),r) in zip(edges2, xinit2)]
n = 1000000
N = 100
k = 1
whichf = "acc"

### Temperature parameter
# betas = 10 .^ range(-5, stop=-2, length=60)
betas = 10 .^ range(-5, stop=2, length=60)
# betas = vcat(range1, range2, range3)
nparams = length(betas)

### Other parameters
# gammas = [1.5, 2.5, 5, 10, 50, 500, 5000, 50000]
gammas = [1,1.5, 2.5, 10]
# gammas = range(1, stop=4, length=10)
# wvals = [1e-5, 2, 5, 10, 50, 500]
wvals = [1,6]

### objective functions

# returns - IR_acc_mix - beta * IR_acc_pure - gamma * EP
function target_acc(logrates, args...)
    edges1, edges2, iedges, cr, dc, on_ind1, on_ind2, beta, gamma, w = args
    logcw = log10(cr) + log10(w)
    kb, ku = logrates
    logrates2 = [kb, ku, kb + logcw, ku + log10(gamma)]

    wedges1 = [(u,v,10 ^ r) for ((u,v),r) in zip(edges1, logrates)]
    wedges2 = [(u,v,10 ^ r) for ((u,v),r) in zip(edges2, logrates2)]

    Q1 = build_mat(wedges1, iedges, cr)
    Q2 = build_mat(wedges2, iedges, cr)
    EP = EP_ss(Q1)

    ir_acc_pure = ir_acc(wedges1, iedges, cr, dc, on_ind1)
    ir_acc_mix = ir_acc(wedges2, iedges, cr, dc, on_ind2)
    # ir_acc_mix = log2((pi2[4])/(pi2[6]))

    (- ir_acc_pure - beta*ir_acc_mix, ir_acc_mix, ir_acc_pure, EP)
end

# returns - IR_full_mix - beta * IR_full_pure - gamma * EP
function target_full(logrates, args...)
    m1, m2, beta, gamma, w, use_mean = args
    logcw = log10(m1.c) + log10(w)
    kb, ku = logrates
    logrates2 = [kb, ku, kb + logcw, ku + log10(gamma)]

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

    ir_full!(m1)
    ir_full!(m2)
    nskip = 5

    # empirical mean estimator
    if use_mean
        ir_full_pure = m1.llr[end] / sum(m1.rt)
        ir_full_mix = m2.llr[end] / sum(m2.rt)
    else

    # variance estimator
        dXs1 = diff(m1.llr[1:nskip:end-1])
        time1 = 2 .* cumsum(m1.rt)
        sq_deltat1 = map(x->sqrt(x), diff(time1[1:nskip:end]))
        ir_full_pure = var(dXs1 ./ sq_deltat1)

        dXs2 = diff(m2.llr[1:nskip:end-1])
        time2 = 2 .* cumsum(m2.rt)
        sq_deltat2 = map(x->sqrt(x), diff(time2[1:nskip:end]))
        ir_full_mix = var(dXs2 ./ sq_deltat2)
    end

    (- ir_full_pure - beta*ir_full_mix, ir_full_mix, ir_full_pure, EP)
end