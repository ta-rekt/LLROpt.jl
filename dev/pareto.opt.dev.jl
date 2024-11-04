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

let
    specfile = ARGS[1]
    cpukey = parse(Int32, ARGS[2])
    ncpus = parse(Int32, ARGS[3])
    outdir = ARGS[4]

    specpath = string("../spec/", specfile)
    outpath = "data/$(outdir)/"
    Base.Filesystem.mkpath(outpath)

    include(specpath)

    ### Optimization starting points
    nparams_per_cpu = div(nparams, ncpus)
    xs = Matrix{Float32}(undef, nedges, nparams_per_cpu)
    xk = Vector{Float32}(undef, nedges)
    xk = x0

    for np in 1:nparams_per_cpu
        for ne in 1:nedges
            xs[ne, np] = x0[ne]
        end
    end

    results_acc = Dict{Int64, Vector{Float32}}()
    results_full = Dict{Int64, Vector{Float32}}()

    ### Main loop
    for n in ns
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
        
        for k in 1:nparams_per_cpu
            keyb = (cpukey - 1) * nparams_per_cpu + k
            beta = betas[keyb]
            for ne in 1:nedges
                xk[ne] = xs[ne, k]
            end

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
                prob_full = Optimization.OptimizationProblem(optf_full, xk, lb=llb, ub=lub, maxiters=maxiters_full)
                sol_full = solve(prob_full, BBO_adaptive_de_rand_1_bin_radiuslimited())

                results_full[keyb] = vcat(sol_full.u, 0, ep_full(sol_full.u), 0)

                irvals = Vector{Float32}(undef, N)
                args_full = [m, beta, true, e]   # use mean estimator for final output
                _ir_full_mean(x) = ir_full_log(x, args_full...)[2]
                xopt = results_full[keyb][1:length(sol_full.u)]
                xs[1:end, k] = sol_full.u

                for i in 1:N
                    irvals[i] = _ir_full_mean(xopt)
                end

                results_full[keyb][end-2] = Statistics.mean(irvals)
                results_full[keyb][end] = Statistics.std(irvals)
            end
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
        params["ns"] = ns
        params["nepochs"] = nepochs
        params["N"] = N
        params["maxiters_acc"] = maxiters_acc
        params["maxiters_full"] = maxiters_full

        paramfile = string(outpath, "params.json")
            
        open(paramfile, "w") do f
            JSON.print(f, params, 4)
        end
    end
end