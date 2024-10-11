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
        pi = P_ss(Q)
        J = pi[u] * Q[u,v] - pi[v] * Q[v,u]
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

let
    specfile = ARGS[1]
    cpukey = parse(Int32, ARGS[2])
    ncpus = parse(Int32, ARGS[3])
    outdir = ARGS[4]

    specpath = string("../spec/", specfile)
    outpath = "data/$(outdir)/"
    Base.Filesystem.mkpath(outpath)

    include(specpath)

    m = LLROptModel(;
        weighted_edges = we,
        input_edges = input_edges,
        on_ind = on_ind,
        c = c,
        dc = dc,
        n = n,
        k = k,
        idist = idist,
    )
    init!(m)
    nparams_per_cpu = div(nparams, ncpus)

    results_acc = Dict{Int64, Vector{Float32}}()
    results_full = Dict{Int64, Vector{Float32}}()
    
    for k in 1:nparams_per_cpu
        keyb = (cpukey - 1) * nparams_per_cpu + k
        beta = betas[keyb]

        if whichf == "acc" || whichf == "both"
            # ACC
            args_acc = [edges, m.input_edges, c, dc, on_ind, beta, e]
            targetfun_acc(x, p) = ir_acc_log(x, args_acc...)[1]
            ir_acc(x) = ir_acc_log(x, args_acc...)[2]
            ep_acc(x) = ir_acc_log(x, args_acc...)[3]

            optf_acc = Optimization.OptimizationFunction(targetfun_acc)
            prob_acc = Optimization.OptimizationProblem(optf_acc, x0, lb=llb, ub=lub, maxiters=maxiters_acc)
            sol_acc = solve(prob_acc, BBO_adaptive_de_rand_1_bin_radiuslimited())

            # RESULTS
            results_acc[keyb] = vcat(sol_acc.u, ir_acc(sol_acc.u), ep_acc(sol_acc.u))
        end

        if whichf == "full" || whichf == "both"
            # FULL
            args_full = [m, beta, false, e]  # use variance estimator in optimization
            targetfun_full(x, p) = ir_full_log(x, args_full...)[1]
            _ir_full(x) = ir_full_log(x, args_full...)[2]
            ep_full(x) = ir_full_log(x, args_full...)[3]

            optf_full = Optimization.OptimizationFunction(targetfun_full)
            prob_full = Optimization.OptimizationProblem(optf_full, x0, lb=llb, ub=lub, maxiters=maxiters_full)
            sol_full = solve(prob_full, BBO_adaptive_de_rand_1_bin_radiuslimited())

            results_full[keyb] = vcat(sol_full.u, 0, ep_full(sol_full.u), 0)

            irvals = Vector{Float32}(undef, N)
            args_full = [m, beta, true, e]   # use mean estimator for final output
            _ir_full_mean(x) = ir_full_log(x, args_full...)[2]
            xopt = results_full[keyb][1:length(sol_full.u)]

            for i in 1:N
                irvals[i] = _ir_full_mean(xopt)
            end

            results_full[keyb][end-2] = Statistics.mean(irvals)
            results_full[keyb][end] = Statistics.std(irvals)
        end
    end

    ######### Write data

    if whichf == "acc" || whichf == "both"
        datafile_acc = string(outpath, "ir_acc_vs_ep_pareto_pid$(cpukey).json")
            
        open(datafile_acc, "w") do f
            JSON.print(f, results_acc, 4)
        end
    end

    if whichf == "full" || whichf == "both"
        datafile_full = string(outpath, "ir_full_vs_ep_pareto_pid$(cpukey).json")

        open(datafile_full, "w") do f
            JSON.print(f, results_full, 4)
        end
    end

    ########## Write parameters

    if cpukey == 1
        params = Dict{String, Any}()
        params["ncpus"] = ncpus
        params["nparams"] = nparams
        params["betas"] = betas
        params["n"] = n
        params["N"] = N
        params["maxiters_acc"] = maxiters_acc
        params["maxiters_full"] = maxiters_full

        paramfile = string(outpath, "params.json")
            
        open(paramfile, "w") do f
            JSON.print(f, params, 4)
        end
    end
end