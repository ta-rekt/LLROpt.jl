import Pkg
Pkg.add("JSON")
Pkg.add("Statistics")
Pkg.add("Optimization")
Pkg.add("OptimizationBBO")
Pkg.add("OptimizationCMAEvolutionStrategy")
using JSON
using Statistics
using Optimization
using OptimizationBBO
using OptimizationCMAEvolutionStrategy
using LLROpt

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
    EP = EP_ss(Q1)
    ir_acc_pure = ir_acc(wedges1, iedges, cr, dc, on_ind1)
    ir_acc_mix = ir_acc(wedges2, iedges, cr, dc, on_ind2)

    (- ir_acc_pure - beta*ir_acc_mix, ir_acc_mix, ir_acc_pure, EP)
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
    ir_full!(m1)
    ir_full!(m2)
    nskip = 5

    # empirical mean estimator
    # ir_full_pure = m1.llr[end] / sum(m1.rt)
    # ir_full_mix = m2.llr[end] / sum(m2.rt)

    # variance estimator
    dXs1 = diff(m1.llr[1:nskip:end-1])
    time1 = 2 .* cumsum(m1.rt)
    sq_deltat1 = map(x->sqrt(x), diff(time1[1:nskip:end]))
    ir_full_pure = var(dXs1 ./ sq_deltat1)

    dXs2 = diff(m2.llr[1:nskip:end-1])
    time2 = 2 .* cumsum(m2.rt)
    sq_deltat2 = map(x->sqrt(x), diff(time2[1:nskip:end]))
    ir_full_mix = var(dXs2 ./ sq_deltat2)

    (- ir_full_pure - beta*ir_full_mix, ir_full_mix, ir_full_pure, EP)
end

let
    keyb = parse(Int32, ARGS[1])
    fname = ARGS[2]

    a = 3e-3        # um
    D_tf = 7.4      # um^2/s
    D_cplx = 0.1    # um^2/s
    w = 0.2
    cr = 5.6         # um^-3
    dc = 0.1 * cr
    cw = w * cr
    dl_tf = 4*a*D_tf
    ub = 5
    lb = 1e-5
    max_ir_full = 0.5 * (dc/cr)^2 * (cr*dl_tf*ub)/(cr*dl_tf + ub)

    # upper bounds for pure ligand circuit
    llb1 = ones(8) .* log10(lb)
    lub1 = ones(8) .* log10(ub)
    lub1[1] = log10(dl_tf)
    lub1[3] = log10(dl_tf)

    # upper bounds for mixed ligands circuit
    llb2 = ones(14) .* log10(lb)
    lub2 = ones(14) .* log10(ub)
    lub2[6] = log10(dl_tf)
    lub2[7] = log10(dl_tf)

    # circuit descriptions
    edges1 = [(1,3), (1,2), (2,4), (2,1), (3,1), (3,4), (4,2), (4,3)]
    edges2 = [(5,1), (1,5), (2,6), (6,2), (1,2), (1,3), (2,4), (2,1), 
              (3,1), (3,4), (4,2), (4,3), (5,6), (6,5)]
    input_edges = [(1,3), (2,4)]
    on_ind1 = BitArray([0,1,0,1])
    on_ind2 = BitArray([0,1,0,1,0,1])

    # GCMAES params
    maxiters_acc = 20000
    maxiters_full = 5000
    xinit1 = (lub1 .+ llb1) ./ 2
    xinit2 = (lub2 .+ llb2) ./ 2
    we1 = [(u,v,r) for ((u,v),r) in zip(edges1, xinit1)]
    we2 = [(u,v,r) for ((u,v),r) in zip(edges2, xinit2)]
    n = 500000

    m1 = LLROptModel(;
        weighted_edges = we1,
        input_edges = input_edges,
        on_ind = on_ind1,
        c = cr,
        dc = dc,
        n = n,
        idist = ones(4) ./ 4,
    )

    m2 = LLROptModel(;
        weighted_edges = we2,
        input_edges = input_edges,
        on_ind = on_ind2,
        c = cr,
        dc = dc,
        n = n,
        idist = ones(6) ./ 6,
    )

    init!(m1)
    init!(m2)

    betas = 10 .^ range(-1.5, stop=1.5, length=15)
    # betas = map(x -> parse(Float64, x), ARGS)
    # gammas = [2.5, 5, 10, 50, 500, 5000, 50000]
    gammas = [5, 50]
    wvals = [10]
    beta = betas[keyb]

    results_acc = Dict{Int64, Dict{Int64, Dict{Int64, Vector{Float32}}}}()
    results_full = Dict{Int64, Dict{Int64, Dict{Int64, Vector{Float32}}}}()

    # for (keyb, beta) in enumerate(betas)

    results_acc[keyb] = Dict{Int64, Dict{Int64, Vector{Float32}}}()
    results_full[keyb] = Dict{Int64, Dict{Int64, Vector{Float32}}}()

    for (keyg, gamma) in enumerate(gammas)
        results_acc[keyb][keyg] = Dict{Int64, Vector{Float32}}()
        results_full[keyb][keyg] = Dict{Int64, Vector{Float32}}()

        for (keyw, w) in enumerate(wvals)

            # ACC
            args_acc = [edges1, edges2, input_edges, cr, dc, on_ind1, on_ind2, 
                beta, gamma, w]
            targetfun_acc(x, p) = target_acc(x, args_acc...)[1]
            ir_acc_mix(x) = target_acc(x, args_acc...)[2]
            ir_acc_pure(x) = target_acc(x, args_acc...)[3]
            ep_acc(x) = target_acc(x, args_acc...)[4]

            optf_acc = Optimization.OptimizationFunction(targetfun_acc)
            prob_acc = Optimization.OptimizationProblem(optf_acc, 
                        xinit1, lb=llb1, ub=lub1, maxiters=maxiters_acc)
            sol_acc = solve(prob_acc, BBO_adaptive_de_rand_1_bin_radiuslimited())

            # FULL
            args_full = [m1, m2, beta, gamma, w]
            targetfun_full(x, p) = target_full(x, args_full...)[1]
            ir_full_mix(x) = target_full(x, args_full...)[2]
            ir_full_pure(x) = target_full(x, args_full...)[3]
            ep_full(x) = target_full(x, args_full...)[4]

            optf_full = Optimization.OptimizationFunction(targetfun_full)
            prob_full = Optimization.OptimizationProblem(optf_full, 
                xinit1, lb=llb1, ub=lub1, maxiters=maxiters_full)
            sol_full = solve(prob_full, BBO_adaptive_de_rand_1_bin_radiuslimited())

            # RESULTS
            results_acc[keyb][keyg][keyw] = vcat(sol_acc.u, ir_acc_mix(sol_acc.u), 
                ir_acc_pure(sol_acc.u), ep_acc(sol_acc.u), beta, gamma, w)
            results_full[keyb][keyg][keyw] = vcat(sol_full.u, 0, 0, 0, 0, 
                ep_full(sol_full.u), beta, gamma, w)
        end
    end
    # end

    N = 100

    # results_full = JSON.parsefile(file2)

    for (keyg, gamma) in enumerate(gammas) 
        for (keyw, w) in enumerate(wvals)
            irvals_mix = Vector{Float32}(undef, N)
            irvals_pure = Vector{Float32}(undef, N)
            args_full = [m1, m2, beta, gamma, w]
            ir_full_mix(x) = target_full(x, args_full...)[2]
            ir_full_pure(x) = target_full(x, args_full...)[3]
            xopt = results_full[keyb][keyg][keyw][1:8]
        
            for i in 1:N
                irvals_mix[i] = ir_full_mix(xopt)
                irvals_pure[i] = ir_full_pure(xopt)
            end

            results_full[keyb][keyg][keyw][9] = Statistics.mean(irvals_mix)
            results_full[keyb][keyg][keyw][10] = Statistics.mean(irvals_pure)
            results_full[keyb][keyg][keyw][11] = Statistics.std(irvals_mix)
            results_full[keyb][keyg][keyw][12] = Statistics.std(irvals_pure)
        end
    end

    path = "data/$(fname)/"
    Base.Filesystem.mkpath(path)
    file1 = string(path, "ir_acc_mix_vs_pure_pareto_beta$(keyb).json")
    file2 = string(path, "ir_full_mix_vs_pure_pareto_beta$(keyb).json")

    # file1 = string(path, "ir_acc_mix_vs_pure_pareto_more_betas.json")
    # file2 = string(path, "ir_full_mix_vs_pure_pareto_more_betas.json")

    open(file1, "w") do f
        JSON.print(f, results_acc, 4)
    end

    open(file2, "w") do f
        JSON.print(f, results_full, 4)
    end

end

