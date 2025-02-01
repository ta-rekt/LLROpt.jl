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
cr = 0.01        # um^-3
dc = 0.1 * cr
dl_tf = 4*a*D_tf
ub = 5
lb = 1e-5

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
# inpath = "/home/tohme/.julia/dev/LLROpt/data/20241118/dr-epsilon-hernan/bigsweep-cmaes-A-pn-smallw/ir_full_mix_vs_pure_pareto_pid"
inpath = ""

### Initialization at midpoint of bounds (trivial)
xinit1 = (lub1 .+ llb1) ./ 2
xinit2 = (lub2 .+ llb2) ./ 2

we1 = [(u,v,r) for ((u,v),r) in zip(edges1, xinit1)]
we2 = [(u,v,r) for ((u,v),r) in zip(edges2, xinit2)]

### Optimization parameters
opt_alg = BBO_adaptive_de_rand_1_bin_radiuslimited()
# opt_alg = Evolutionary.CMAES(μ = 5, λ = 10, sigma0 = 1)

maxiters_acc = 30000
maxiters_full = 1500
n = 40000000
N = 100
k = 1
whichf = "full"

### Temperature parameter
angle_incr = 3.141592 / 1e3
# betasn = map(x->-tan(x), range(0*angle_incr, stop=0.5*angle_incr, length=128))
# betasp = map(x->tan(x), range(0*angle_incr, stop=1*angle_incr, length=128))
# betas = vcat(betasn, betasp)
betas = 10 .^ range(-2, stop=4, length=64)
nparams = length(betas)

### Other parameters
gammas = [10]
wvals = [1, 10, 100]

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
    cr = beta
    dc = cr / 10
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
    P = 1e-4 - (pi2[4] + pi2[6])
    Pplus = 10000*(P + abs(P))   # penalty for transcription rate falling below 1e-5

    ir_acc_pure = ir_acc(wedges1, iedges, cr, dc, on_ind1)
    ir_acc_mix = ir_acc(wedges2, iedges, cr, dc, on_ind2)
    mlog_epsilon = log2(w*pi2[4]/pi2[6])

    # (- ir_acc_pure - beta*ir_acc_mix + Jplus, ir_acc_mix, ir_acc_pure, EP)
    # (- ir_acc_mix - beta*mlog_epsilon + Jplus, mlog_epsilon, ir_acc_mix, EP)
    (- ir_acc_mix + Jplus + Pplus, mlog_epsilon, ir_acc_mix, EP)
    
end

# returns - IR_full_mix - beta * IR_full_pure - gamma * EP
@counted function target_full(logrates, args...)
    m1, m2, beta, gamma, w = args

    m1.c = beta
    m2.c = beta
    m1.dc = beta / 10
    m2.dc = beta / 10

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

    sigma = 5e-8
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
    ir_full!(m2)

    # empirical mean estimator
    # ir_full_pure = m1.llr[end] / sum(m1.rt)
    ir_full_mix = m2.llr[end] / sum(m2.rt)
    # ir_full_mix = ir_acc(m2.weighted_edges, m2.input_edges, m2.c, m2.dc, m2.on_ind) + rand(e)
    mlog_epsilon = log2(w*(pi2[4])/(pi2[6]))

    # (- ir_full_pure - beta*ir_full_mix + Jplus, ir_full_mix, ir_full_pure, EP)
    # (- ir_full_mix - beta*mlog_epsilon + Jplus, mlog_epsilon, ir_full_mix, EP)
    (- ir_full_mix + Jplus, mlog_epsilon, ir_full_mix, EP)
end



### Local search around circuit A
# xinit1 = [-1.0515872,
#           -4.9898477,
#           -1.0515872,
#           0.69897,
#           0.2825698,
#           0.69897,
#           0.46107602,
#           -4.999996]

# xinit2 = [1.2825698,
#           0.09454083567823801,
#           0.09454083567823801,
#           1.46107602,
#           -4.9898477,
#           -1.0515872,
#           -1.0515872,
#           0.69897,
#           0.2825698,
#           0.69897,
#           0.46107602,
#           -4.999996,
#           0.69897,
#           -4.999996]

# llb1[3] = log10(dl_tf) - 0.05

### Local search around circuit B
# xinit2 = [1.6989699,
#         0.09454083567823801,
#         -3.853808964321762,
#         1.54783356,
#         -4.989879,
#         -1.0515872,
#         -4.999937,
#         0.69897,
#         0.6989699,
#         0.69896984,
#         0.54783356,
#         -4.9945397,
#         0.69896984,
#         -4.9945397]

# xinit1 = [-1.0515872,
#         -4.989879,
#         -4.999937,
#         0.69897,
#         0.6989699,
#         0.69896984,
#         0.54783356,
#         -4.9945397]

# lub1[3] = -3

### Local search around circuit C
# xinit1 = [-1.0515921,
#         0.6989517,
#         -4.999785,
#         0.69896877,
#         -0.12840532,
#         -1.9895186,
#         0.69896877,
#         -4.9985023]

# xinit2 = [0.2695346886720376,
#         0.09453593567823804,
#         -3.853656964321762,
#         1.0969087786720375,
#         0.6989517,
#         -1.0515921,
#         -4.999785,
#         0.69896877,
#         -0.12840532,
#         -1.9895186,
#         0.69896877,
#         -4.9985023,
#         -1.9895186,
#         -4.9985023]

# llb1[2] = log10(ub) - 0.05
# lub1[3] = -4.95
