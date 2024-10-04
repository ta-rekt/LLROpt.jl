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
    pi = P_ss(Q1)
    pi2 = P_ss(Q2)
    J = pi[1] * Q1[1,2] - pi[2] * Q1[2,1]
    Jplus = J + abs(J)   # penalty for counterclockwise flux
    ir_acc_pure = ir_acc(wedges1, iedges, cr, dc, on_ind1)
    # ir_acc_mix = ir_acc(wedges2, iedges, cr, dc, on_ind2)
    ir_acc_mix = log2((pi2[4])/(pi2[6]))

    (- ir_acc_pure - beta*ir_acc_mix + Jplus, ir_acc_mix, ir_acc_pure, EP)
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
    pi = P_ss(m.Q_c)
    J = pi[1] * m.Q_c[1,2] - pi[2] * m.Q_c[2,1]
    Jplus = J + abs(J)   # penalty for counterclockwise flux
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

    (- ir_full_pure - beta*ir_full_mix + Jplus, ir_full_mix, ir_full_pure, EP)
end

let
    specfile = ARGS[1]
    cpukey = parse(Int32, ARGS[2])
    ncpus = parse(Int32, ARGS[3])
    outdir = ARGS[4]

    specpath = string("../spec/", specfile)
    outpath = "data/$(outdir)/"
    Base.Filesystem.mkpath(outpath)

    include(specpath)

    m1 = LLROptModel(;
        weighted_edges = we1,
        input_edges = input_edges,
        on_ind = on_ind1,
        c = cr,
        dc = dc,
        n = n,
        k = k,
        idist = idist1,
    )

    m2 = LLROptModel(;
        weighted_edges = we2,
        input_edges = input_edges,
        on_ind = on_ind2,
        c = cr,
        dc = dc,
        n = n,
        k = k,
        idist = idist2,
    )

    init!(m1)
    init!(m2)

    nparams_per_cpu = div(nparams, ncpus)

    irvals_mix = Vector{Float32}(undef, N)
    irvals_pure = Vector{Float32}(undef, N)
    results_acc = Dict{Int64, Dict{Int64, Dict{Int64, Vector{Float32}}}}()
    results_full = Dict{Int64, Dict{Int64, Dict{Int64, Vector{Float32}}}}()

    for k in 1:nparams_per_cpu
        keyb = (cpukey - 1) * nparams_per_cpu + k
        beta = betas[keyb]

        # IR ACC
        if whichf == "acc" || whichf == "both"
            results_acc[keyb] = Dict{Int64, Dict{Int64, Vector{Float32}}}()

            for (keyg, gamma) in enumerate(gammas)
                results_acc[keyb][keyg] = Dict{Int64, Vector{Float32}}()
        
                for (keyw, w) in enumerate(wvals)
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

                    # RESULTS
                    results_acc[keyb][keyg][keyw] = vcat(sol_acc.u, ir_acc_mix(sol_acc.u), 
                        ir_acc_pure(sol_acc.u), ep_acc(sol_acc.u), beta, gamma, w)
                end
            end
        end

        # IR FULL
        if whichf == "full" || whichf == "both"
            results_full[keyb] = Dict{Int64, Dict{Int64, Vector{Float32}}}()

            for (keyg, gamma) in enumerate(gammas)
                results_acc[keyb][keyg] = Dict{Int64, Vector{Float32}}()
                results_full[keyb][keyg] = Dict{Int64, Vector{Float32}}()

                for (keyw, w) in enumerate(wvals)
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
                    results_full[keyb][keyg][keyw] = vcat(sol_full.u, 0, 0, 0, 0, 
                        ep_full(sol_full.u), beta, gamma, w)

                    # args_full = [m1, m2, beta, gamma, w]
                    # ir_full_mix(x) = target_full(x, args_full...)[2]
                    # ir_full_pure(x) = target_full(x, args_full...)[3]
                    # xopt = results_full[keyb][keyg][keyw][1:8]
                
                    for i in 1:N
                        irvals_mix[i] = ir_full_mix(sol_full.u)
                        irvals_pure[i] = ir_full_pure(sol_full.u)
                    end

                    results_full[keyb][keyg][keyw][9] = Statistics.mean(irvals_mix)
                    results_full[keyb][keyg][keyw][10] = Statistics.mean(irvals_pure)
                    results_full[keyb][keyg][keyw][11] = Statistics.std(irvals_mix)
                    results_full[keyb][keyg][keyw][12] = Statistics.std(irvals_pure)
                end
            end
        end
    end

    ######### Write data

    if whichf == "acc" || whichf == "both"
        file1 = string(outpath, "ir_acc_mix_vs_pure_pareto_pid$(cpukey).json")

        open(file1, "w") do f
            JSON.print(f, results_acc, 4)
        end
    end

    if whichf == "full" || whichf == "both"
        file2 = string(outpath, "ir_full_mix_vs_pure_pareto_pid$(cpukey).json")

        open(file2, "w") do f
            JSON.print(f, results_full, 4)
        end
    end

    ########## Write parameters

    if cpukey == 1
        if whichf == "acc" || whichf == "both"
            params_acc = Dict{String, Any}()
            params_acc["ncpus"] = ncpus
            params_acc["nparams"] = nparams
            params_acc["betas"] = betas
            params_acc["gammas"] = gammas
            params_acc["wvals"] = wvals

            paramfile_acc = string(outpath, "params_acc.json")
                
            open(paramfile_acc, "w") do f
                JSON.print(f, params_acc, 4)
            end
        end

        if whichf == "full" || whichf == "both"
            params_full = Dict{String, Any}()
            params_full["ncpus"] = ncpus
            params_full["nparams"] = nparams
            params_full["betas"] = betas
            params_full["gammas"] = gammas
            params_full["wvals"] = wvals

            paramfile_full = string(outpath, "params_full.json")

            open(paramfile_full, "w") do f
                JSON.print(f, params_full, 4)
            end
        end
    end
end
