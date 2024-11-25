import Pkg
Pkg.activate("LLROpt")
using JSON
using LLROpt
using Statistics

let
    specfile = ARGS[1]
    cpukey = parse(Int32, ARGS[2])
    ncpus = parse(Int32, ARGS[3])
    outdir = ARGS[4]

    # output directory
    outpath = "data/$(outdir)/"
    Base.Filesystem.mkpath(outpath)

    # specification file
    specpath = string("../spec/", specfile)
    include(specpath)

    # input data
    if inpath !== ""
        infile = string(inpath, cpukey, ".json")
        results_init = Dict()
        open(infile, "r") do f
            results_init = JSON.parse(f)
        end
    end

    results_acc = Dict{Int64, Vector{Float32}}()
    results_full = Dict{Int64, Vector{Float32}}()

    nparams_per_cpu = div(nparams, ncpus)

    ### Main loop
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
        
        if whichf == "acc" || whichf == "both"
            if inpath !== ""
                xinit = results_init[string(keyb)][1:end-2]
                xinit = Vector{Float64}(xinit)
            else
                xinit = xinit1
            end

            # ACC
            args_acc = [edges, m.input_edges, c, dc, on_ind, beta, e]
            targetfun_acc(x, p) = ir_acc_log(x, args_acc...)[1]
            ir_acc(x) = ir_acc_log(x, args_acc...)[2]
            ep_acc(x) = ir_acc_log(x, args_acc...)[3]

            optf_acc = Optimization.OptimizationFunction(targetfun_acc)
            prob_acc = Optimization.OptimizationProblem(optf_acc, xinit, lb=llb, ub=lub, maxiters=maxiters_acc)
            sol_acc = solve(prob_acc, BBO_adaptive_de_rand_1_bin_radiuslimited())

            # RESULTS
            results_acc[keyb] = vcat(sol_acc.u, ir_acc(sol_acc.u), ep_acc(sol_acc.u))
        end

        if whichf == "full" || whichf == "both"
            if inpath !== ""
                xinit = results_init[string(keyb)][1:end-3]
                xinit = Vector{Float64}(xinit)
            else
                xinit = xinit1
            end

            # FULL
            args_full = [m, beta, e]
            targetfun_full(x, p) = ir_full_log(x, args_full...)[1]
            _ir_full(x) = ir_full_log(x, args_full...)[2]
            ep_full(x) = ir_full_log(x, args_full...)[3]

            optf_full = Optimization.OptimizationFunction(targetfun_full)
            prob_full = Optimization.OptimizationProblem(optf_full, xinit, lb=llb, ub=lub, maxiters=maxiters_full)
            sol_full = solve(prob_full, BBO_adaptive_de_rand_1_bin_radiuslimited())

            results_full[keyb] = vcat(sol_full.u, 0, ep_full(sol_full.u), 0)

            irvals = Vector{Float32}(undef, N)
            args_full = [m, beta, e] 
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
        params["N"] = N
        params["maxiters_acc"] = maxiters_acc
        params["maxiters_full"] = maxiters_full
        params["e"] = e

        paramfile = string(outpath, "params.json")
            
        open(paramfile, "w") do f
            JSON.print(f, params, 4)
        end
    end
end