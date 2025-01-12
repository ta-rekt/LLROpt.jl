using Optimization
using OptimizationBBO
using OptimizationCMAEvolutionStrategy
using OptimizationEvolutionary
using OptimizationManopt
using Distributions

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

### Initialization from stored data
# inpath = "/home/tohme/.julia/dev/LLROpt/data/20241125/ir-dr-hernan/coarse-sweep-1-alg=de-start=MGR-lg=global-whichf=simfull-maxiters=3000-sigma=3e-5-N=3000-nparams=1/ir_full_mix_vs_pure_pareto_pid"
inpath = ""
# inpath = "prevbeta"

### Initialization at midpoint of bounds (trivial)
xinit1 = (lub1 .+ llb1) ./ 2
xinit2 = (lub2 .+ llb2) ./ 2

we1 = [(u,v,r) for ((u,v),r) in zip(edges1, xinit1)]
we2 = [(u,v,r) for ((u,v),r) in zip(edges2, xinit2)]

### Optimization parameters
opt_alg = BBO_adaptive_de_rand_1_bin_radiuslimited()
# opt_alg = Evolutionary.CMAES(μ = 5, λ = 10, sigma0 = 0.01)

maxiters_acc = 30000
maxiters_full = 1000
n = 100000
N = 1
k = 1
whichf = "acc"

### temperature parameter
betas = 10 .^ range(-1, stop=3, length=64)
nparams = length(betas)

### Other parameters
gammas = [1.5, 2.5, 10]
wvals = [2.5, 100]

### objective functions

const COUNTERS = Dict{String, Int}()

macro counted(f)
    name = f.args[1].args[1]
    name_str = String(name)
    body = f.args[2]
    counter_code = quote
        if !haskey(COUNTERS, $name_str)
            COUNTERS[$name_str] = 0
        end
        COUNTERS[$name_str] += 1
    end
    insert!(body.args, 1, counter_code)
    return f
end

# returns - IR_acc_mix - beta * IR_acc_pure - gamma * EP
@counted function target_acc(logrates, args...)
    edges1, edges2, iedges, cr, dc, on_ind1, on_ind2, beta, gamma, w = args
    logcw = log10(beta) + log10(w)
    kboff, konu, kbon, koffu, kuoff, konb, kuon, koffb = logrates
    logrates2 = [kuoff+log10(gamma), kboff+logcw, kbon+logcw, kuon+log10(gamma), 
        konu, kboff, kbon, koffu, kuoff, konb, kuon, koffb, konb, koffb]

    wedges1 = [(u,v,10 ^ r) for ((u,v),r) in zip(edges1, logrates)]
    wedges2 = [(u,v,10 ^ r) for ((u,v),r) in zip(edges2, logrates2)]

    Q1 = build_mat(wedges1, iedges, beta)
    Q2 = build_mat(wedges2, iedges, beta)
    # EP = EP_ss(Q1)
    EP = EP_ss(Q2)

    Jplus = 0
    pi = P_ss(Q1)
    pi2 = P_ss(Q2)
    J = pi[1] * Q1[1,2] - pi[2] * Q1[2,1]
    Jplus = J + abs(J)   # penalty for counterclockwise flux
    ir_acc_pure = ir_acc(wedges1, iedges, beta, beta/10, on_ind1)
    ir_acc_mix = ir_acc(wedges2, iedges, beta, beta/10, on_ind2)
    # mlog_epsilon = log2((pi2[4])/(pi2[6]))

    # (- ir_acc_pure - beta*ir_acc_mix + Jplus, ir_acc_mix, ir_acc_pure, EP)
    (- ir_acc_mix + Jplus, ir_acc_mix, beta, EP)
    
end

# returns - IR_full_mix - beta * IR_full_pure - gamma * EP
@counted function target_full(logrates, args...)
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

    sigma = 1e-10
    e = Normal(0, sigma)

    init!(m1)
    init!(m2)
    EP = EP_ss(m2.Q_c)

    Jplus = 0
    pi = P_ss(m1.Q_c)
    pi2 = P_ss(m2.Q_c)
    J = pi[1] * m1.Q_c[1,2] - pi[2] * m1.Q_c[2,1]
    Jplus = J + abs(J)   # penalty for counterclockwise flux
    # ir_full!(m1)
    # ir_full!(m2)

    # empirical mean estimator
    # ir_full_pure = m1.llr[end] / sum(m1.rt)
    # ir_full_mix = m2.llr[end] / sum(m2.rt)
    ir_full_pure = ir_acc(m1.weighted_edges, m1.input_edges, m1.c, m1.dc, m1.on_ind) + rand(e)
    ir_full_mix = ir_acc(m2.weighted_edges, m2.input_edges, m2.c, m2.dc, m2.on_ind) + rand(e)
    # mlog_epsilon = log2((pi2[4])/(pi2[6]))

    (- ir_full_pure - beta*ir_full_mix + Jplus, ir_full_mix, ir_full_pure, EP)
    # (- ir_full_mix - beta*mlog_epsilon + Jplus, mlog_epsilon, ir_full_mix, EP)
end
