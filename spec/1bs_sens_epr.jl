using Optimization
using OptimizationBBO
using OptimizationGCMAES
using OptimizationOptimJL

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
# inpath = "/home/tohme/.julia/dev/LLROpt/data/20241113/ir-dr/ir_full_mix_vs_pure_pareto_pid"
inpath = ""
maxiters_acc = 30000
maxiters_full = 2000
sigma0 = log10(ub) / 4
xinit1 = (lub .+ llb) ./ 2
we = [(u,v,r) for ((u,v),r) in zip(edges, xinit1)]

k = 1
N = 5000
nepochs = 0
startexp = 20
ns = [2^i for i in range(startexp, stop=startexp+nepochs)]
whichf = "full"

### temperature parameter
betas = vcat(range(3e-4, stop=1e-5, length=10), 10 .^ range(-4.8, stop=-9, length=5))
nparams = length(betas)

# IR ACC
function ir_acc_log(logrates, args...)
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
    EP = EP_ss(Q)
    
    # clockwise bias by forcing flow along edge e
    Jplus = 0
    m, n = e
    if m >= 0
        pi = P_ss(Q)
        J = pi[m] * Q[m,n] - pi[n] * Q[n,m]
        Jplus = J + abs(J)
    end

    IR = ir_acc(wedges, iedges, c, dc, on_ind)

    (- IR + beta*EP + Jplus, IR, EP)
end

# IR FULL
function ir_full_log(logrates, args...)
    m, beta, use_mean, e = args

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
    if use_mean
        IR = m.llr[end] / sum(m.rt)
    
    # variance estimator
    else
        nskip = 5
        dXs = diff(m.llr[1:nskip:end-1])
        time = 2 .* cumsum(m.rt)
        sq_deltat = map(x->sqrt(x), diff(time[1:nskip:end]))
        IR = var(dXs ./ sq_deltat)
    end

    (- IR + beta*EP + Jplus, IR, EP)
end