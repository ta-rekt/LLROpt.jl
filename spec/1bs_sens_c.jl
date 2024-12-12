using Optimization
using OptimizationBBO
using OptimizationEvolutionary

### biophysical parameters
a = 3e-3        # ums
D_tf = 7.4      # um^2/s
D_cplx = 0.1    # um^2/s
c = 5.6         # um^-3
dc = 0.1 * c
dl_tf = 4*a*D_tf
ub = 5
lb = 1e-5
max_ir_full = 0.5 * (dc/c)^2 * c*4*a*D_tf

### 1-binding site circuit parameters
nedges = 8
edges = [(1,3), (1,2), (2,4), (2,1), (3,1), (3,4), (4,2), (4,3)]
input_edges = [(1,3), (2,4)]
on_ind = BitArray([0,1,0,1])
idist = ones(4) ./ 4
e = (1,2)                   # symmetry-breaking edge

### Bounding box on rates
llb = ones(nedges) .* log10(lb)
lub = ones(nedges) .* log10(ub)
lub[1] = log10(dl_tf)
lub[3] = log10(dl_tf)

### Optimization parameters
# inpath = "/home/tohme/.julia/dev/LLROpt/data/20241107/ir-ep/ir_full_vs_ep_pareto_pid"
inpath = ""

opt_alg = BBO_adaptive_de_rand_1_bin_radiuslimited()
# opt_alg = Evolutionary.CMAES(μ = 5, λ = 10, sigma0 = 0.05)

maxiters_acc = 30000
maxiters_full = 200
sigma0 = log10(ub) / 4
xinit1 = (lub .+ llb) ./ 2
we = [(u,v,r) for ((u,v),r) in zip(edges, xinit1)]

k = 1
N = 10
n = 80000000
whichf = "acc"

### temperature parameter
betas = 10 .^ range(0, stop=2, length=64)
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
    rates = 10 .^ logrates
    wedges = Vector{Tuple{Int64, Int64, Float32}}(undef, nrates)

    for i in 1:nrates
        w1 = edges[i][1]
        w2 = edges[i][2]
        w3 = rates[i]

        wedges[i] = (w1, w2, w3)
    end

    # wedges = [(u,v,10 ^ r) for ((u,v),r) in zip(edges, logrates)]

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

    for (keyb, val) in enumerate(logrates)
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

    # mean estimator
    IR = m.llr[end] / sum(m.rt)

    (- IR + beta*EP + Jplus, IR, EP)
end