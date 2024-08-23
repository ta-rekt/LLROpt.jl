##. - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# Optimization MWE with ir_acc and ir_full
import Pkg
Pkg.add("Optimization")
Pkg.add("OptimizationBBO")
Pkg.add("OptimizationGCMAES")
Pkg.add("OptimizationOptimJL")
Pkg.add("Statistics")
Pkg.add("JSON")
using JSON
using LLROpt
using Optimization
using OptimizationBBO
using OptimizationGCMAES
using OptimizationOptimJL
using Statistics

# IR ACC
function ir_acc_log(logrates, args...)
    edges, iedges, c, dc, on_ind, beta = args
    nrates = length(edges)
    rates = 10 .^ logrates
    wedges = Vector{Tuple{Int64, Int64, Float32}}(undef, nrates)

    for i in 1:nrates
        w1 = edges[i][1]
        w2 = edges[i][2]
        w3 = rates[i]

        wedges[i] = (w1, w2, w3)
    end

    Q = build_mat(wedges, iedges, c)
    EP = EP_ss(Q)
    IR = ir_acc(wedges, iedges, c, dc, on_ind)

    (- IR + beta*EP, IR, EP)
end

# IR FULL
function ir_full_log(logrates, args...)

    m, beta, use_mean = args

    for (key, val) in enumerate(logrates)
        (u,v,r) = m.weighted_edges[key]
        m.weighted_edges[key] = (u,v,10 ^ val)
    end

    init!(m)
    EP = EP_ss(m.Q_c)
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

    (- IR + beta*EP, IR, EP)
end

let
    key = parse(Int32, ARGS[1])
    fname = ARGS[2]

    # rate bounds
    a = 3e-3        # ums
    D_tf = 7.4      # um^2/s
    D_cplx = 0.1    # um^2/s
    c = 5.6         # um^-3
    dc = 0.1 * c
    dl_tf = 4*a*D_tf
    ub = 5
    lb = 1e-5
    max_ir_full = 0.5 * (dc/c)^2 * c*4*a*D_tf

    llb = ones(8) .* log10(lb)
    lub = ones(8) .* log10(ub)
    lub[1] = log10(dl_tf)
    lub[3] = log10(dl_tf)

    # 2-state circuit
    # edges = [(1,2),(2,1)]
    # input_edges = [(1,2)]
    # on_ind = BitArray([0,1])

    # 4-state circuit
    edges = [(1,3), (1,2), (2,4), (2,1), (3,1), (3,4), (4,2), (4,3)]
    input_edges = [(1,3), (2,4)]
    on_ind = BitArray([0,1,0,1])

    # Optimization params
    maxiters_acc = 20000
    maxiters_full = 5000
    sigma0 = log10(ub) / 4
    x0 = (lub .+ llb) ./ 2
    we = [(u,v,r) for ((u,v),r) in zip(edges, x0)]

    m = LLROptModel(;
        weighted_edges = we,
        input_edges = input_edges,
        on_ind = on_ind,
        c = c,
        dc = dc,
        n = 1000000,
        idist = ones(4) ./ 4,
    )
    init!(m)

    betas = vcat(range(3e-4, stop=1e-5, length=10), 10 .^ range(-4.8, stop=-8, length=5))
    # betas = [0]
    # betas = range(2e-4, stop=1e-5, length=5)
    results_acc = Dict{Int64, Vector{Float32}}()
    results_full = Dict{Int64, Vector{Float32}}()
    
    beta = betas[key]

    # ACC
    args_acc = [edges, m.input_edges, c, dc, on_ind, beta]
    targetfun_acc(x, p) = ir_acc_log(x, args_acc...)[1]
    ir_acc(x) = ir_acc_log(x, args_acc...)[2]
    ep_acc(x) = ir_acc_log(x, args_acc...)[3]

    optf_acc = Optimization.OptimizationFunction(targetfun_acc)
    prob_acc = Optimization.OptimizationProblem(optf_acc, x0, lb=llb, ub=lub, maxiters=maxiters_acc)
    sol_acc = solve(prob_acc, BBO_adaptive_de_rand_1_bin_radiuslimited())

    # FULL
    args_full = [m, beta, false]  # use variance estimator in optimization
    targetfun_full(x, p) = ir_full_log(x, args_full...)[1]
    _ir_full(x) = ir_full_log(x, args_full...)[2]
    ep_full(x) = ir_full_log(x, args_full...)[3]

    optf_full = Optimization.OptimizationFunction(targetfun_full)
    prob_full = Optimization.OptimizationProblem(optf_full, x0, lb=llb, ub=lub, maxiters=maxiters_full)
    sol_full = solve(prob_full, BBO_adaptive_de_rand_1_bin_radiuslimited())

    # RESULTS
    results_acc[key] = vcat(sol_acc.u, ir_acc(sol_acc.u), ep_acc(sol_acc.u))
    results_full[key] = vcat(sol_full.u, 0, ep_full(sol_full.u), 0)

    N = 2000
    
    irvals = Vector{Float32}(undef, N)
    args_full = [m, beta, true]   # use mean estimator for final output
    _ir_full(x) = ir_full_log(x, args_full...)[2]
    xopt = results_full[key][1:8]
    # xopt = [dl_tf, ub]

    for i in 1:N
        irvals[i] = _ir_full(xopt)
    end

    results_full[key][9] = Statistics.mean(irvals)
    results_full[key][11] = Statistics.std(irvals)

    path = "data/$(fname)/"
    Base.Filesystem.mkpath(path)
    file1 = string(path, "ir_acc_vs_ep_pareto_4s_beta$(key).json")
    file2 = string(path, "ir_full_vs_ep_pareto_4s_beta$(key).json")

    open(file1, "w") do f
        JSON.print(f, results_acc, 4)
    end

    open(file2, "w") do f
        JSON.print(f, results_full, 4)
    end
end

