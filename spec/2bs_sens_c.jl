using Optimization
using OptimizationBBO
using OptimizationEvolutionary
using LLROpt

### Biophysical parameters
a = 3e-3        # um
D_tf = 7.4      # um^2/s
D_cplx = 0.1    # um^2/s
c = 5.6        # um^-3
dc = 0.1 * c
dl_tf = 4*a*D_tf
ub = 5
lb = 1e-3
# max_ir_full = 0.5 * (dc/cr)^2 * (cr*dl_tf*ub)/(cr*dl_tf + ub)

### Circuit descriptions
nrates = 10
edges = [(5,1), (1,5), (2,6), (6,2), (1,2), (1,3), (2,4),
         (2,1), (3,1), (3,4), (4,2), (4,3), (5,6), (6,5)]
input_edges = [(1,3), (2,4), (5,1), (6,2)]
on_ind = BitArray([0,1,0,1,0,1])
idist = ones(6) ./ 6
e = (5,6)                   # symmetry-breaking edge

### Upper bounds for 2bs circuit
llb = ones(10) .* log10(lb)
lub = ones(10) .* log10(ub)
llb2 = ones(14) .* log10(lb)
lub2 = ones(14) .* log10(ub)
lub[1] = log10(dl_tf)
lub[3] = log10(dl_tf)

### Optimization parameters
# inpath = "/home/tohme/.julia/dev/LLROpt/data/20241111/2bs-ir-ep/ir_full_vs_ep_pareto_pid"
inpath = ""

opt_alg = BBO_adaptive_de_rand_1_bin_radiuslimited()
# opt_alg = Evolutionary.CMAES(μ = 5, λ = 10, sigma0 = 2)

maxiters_acc = 20000
maxiters_full = 3
xinit1 = (lub .+ llb) ./ 2
xinit2 = (lub2 .+ llb2) ./ 2
we = [(u,v,r) for ((u,v),r) in zip(edges, xinit2)]
k = 1
n = 1000
N = 10
whichf = "acc"

### temperature parameter
betas = 10 .^ range(-2, stop=4, length=64)
nparams = length(betas)

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

# IR ACC
@counted function ir_acc_log(logrates, args...)
    edges, iedges, c, dc, on_ind, beta, e = args
    nrates = length(edges)
    kboff, konu, kbon, koffu, kuoff, kon1b, kuon, koff1b, kon2b, koff2b = logrates

    # edges = [(5,1),         (1,5),         (2,6),  (6,2),     (1,2),  (1,3),  (2,4),
    #          (2,1),         (3,1),         (3,4),  (4,2),     (4,3),  (5,6),  (6,5)]

    logrates2 = [kboff+log10(2), kuoff,        kuon,  kbon+log10(2), kon1b, kboff, kbon,
                 koff1b,       kuoff+log10(2), kon2b, kuon+log10(2), koff2b, konu,  koffu]

    rates = 10 .^ logrates2
    wedges = Vector{Tuple{Int64, Int64, Float32}}(undef, nrates)

    for i in 1:nrates
        w1 = edges[i][1]
        w2 = edges[i][2]
        w3 = rates[i]

        wedges[i] = (w1, w2, w3)
    end

    Q = build_mat(wedges, iedges, c)

    # clockwise bias by forcing flow along edge e
    Jplus = 0
    m, n = e
    if m >= 0
        pi = P_ss(Q)
        J = pi[m] * Q[m,n] - pi[n] * Q[n,m]
        Jplus = J + abs(J)
    end

    IR = ir_acc(wedges, iedges, beta, beta/10, on_ind)

    (- IR + Jplus, IR, beta)
end

# IR FULL
@counted function ir_full_log(logrates, args...)
    m, beta, e = args
    kboff, konu, kbon, koffu, kuoff, kon1b, kuon, koff1b, kon2b, koff2b = logrates

    # edges = [(5,1),         (1,5),         (2,6),  (6,2),     (1,2),  (1,3),  (2,4),
    #          (2,1),         (3,1),         (3,4),  (4,2),     (4,3),  (5,6),  (6,5)]

    logrates2 = [kboff+log10(2), kuoff,        kuon,  kbon+log10(2), kon1b, kboff, kbon,
                 koff1b,       kuoff+log10(2), kon2b, kuon+log10(2), koff2b, konu,  koffu]

    for (keyb, val) in enumerate(logrates2)
        (u,v,r) = m.weighted_edges[keyb]
        m.weighted_edges[keyb] = (u,v,10 ^ val)
    end

    init!(m)
    EP = EP_ss(m.Q_c)

    # clockwise bias
    Jplus = 0
    u, v = e
    if u >= 0
        pi = P_ss(m.Q_c)
        J = pi[u] * m.Q_c[u,v] - pi[v] * m.Q_c[v,u]
        Jplus = J + abs(J)
    end

    ir_full!(m)
    # sigma = 1e-5
    # e = Normal(0, sigma)

    # mean estimator
    IR = m.llr[end] / sum(m.rt)
    # IR = ir_acc(m.weighted_edges, m.input_edges, m.c, m.dc, m.on_ind) + rand(e)

    (- IR + beta*EP + Jplus, IR, EP)
end
