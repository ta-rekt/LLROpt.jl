import Pkg
# Pkg.activate("LLROpt")
using JSON
using Statistics
using LLROpt

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
    if inpath !== "" && inpath !== "prevbeta"
        infile = string(inpath, cpukey, ".json")
        results_init = Dict()
        open(infile, "r") do f
            results_init = JSON.parse(f)
        end
    end

    nparams_per_cpu = div(nparams, ncpus)

    irvals_mix = Vector{Float32}(undef, N)
    irvals_pure = Vector{Float32}(undef, N)
    results_acc = Dict{Int64, Dict{Int64, Dict{Int64, Vector{Float32}}}}()
    results_full = Dict{Int64, Dict{Int64, Dict{Int64, Vector{Float32}}}}()

    for k in 1:nparams_per_cpu
        keyb = (cpukey - 1) * nparams_per_cpu + k
        beta = betas[keyb]

        if !fixed_conc
            llb_clipped = max(log10(lb), log10(lb) - log10(beta))
            # lub_clipped = min(log10(dl_tf), log10(ub) - log10(beta))

            llb1[1] = llb_clipped
            llb1[4] = llb_clipped
            # lub1[1] = lub_clipped
            # lub1[4] = lub_clipped

            xinit1 = (lub1 .+ llb1) ./ 2
        end

        # IR ACC
        if whichf == "acc" || whichf == "both"
            results_acc[keyb] = Dict{Int64, Dict{Int64, Vector{Float32}}}()

            for (keyg, gamma) in enumerate(gammas)
                results_acc[keyb][keyg] = Dict{Int64, Vector{Float32}}()
        
                for (keyw, w) in enumerate(wvals)
                    if inpath === ""
                        xinit = xinit1
                    elseif inpath === "prevbeta"
                        if keyb == 1
                            xinit = xinit1
                        else
                            xinit = results_acc[keyb-1][keyg][keyw][1:end-6]
                            xinit = Vector{Float64}(xinit)
                        end
                    else
                        xinit = results_init[string(keyb)][string(keyg)][string(keyw)][1:end-6]
                        xinit = Vector{Float64}(xinit)
                    end

                    args_acc = [edges1, edges2, input_edges, cr, dc, on_ind1, on_ind2, 
                        beta, gamma, w]
                    targetfun_acc(x, p) = target_acc(x, args_acc...)[1]
                    ir_acc_mix(x) = target_acc(x, args_acc...)[2]
                    ir_acc_pure(x) = target_acc(x, args_acc...)[3]
                    ep_acc(x) = target_acc(x, args_acc...)[4]

                    optf_acc = Optimization.OptimizationFunction(targetfun_acc)
                    prob_acc = Optimization.OptimizationProblem(optf_acc, 
                                xinit, lb=llb1, ub=lub1, maxiters=maxiters_acc)
                    sol_acc = solve(prob_acc, opt_alg)

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
                results_full[keyb][keyg] = Dict{Int64, Vector{Float32}}()

                for (keyw, w) in enumerate(wvals)
                    if inpath === ""
                        xinit = xinit1
                    elseif inpath === "prevbeta"
                        if keyb == 1
                            xinit = xinit1
                        else
                            xinit = results_full[keyb-1][keyg][keyw][1:end-8]
                            xinit = Vector{Float64}(xinit)
                        end
                    else
                        xinit = results_init[string(keyb)][string(keyg)][string(keyw)][1:end-8]
                        xinit = Vector{Float64}(xinit)
                    end

                    args_full = [m1, m2, beta, gamma, w]
                    targetfun_full(x, p) = target_full(x, args_full...)[1]
                    ir_full_mix(x) = target_full(x, args_full...)[2]
                    ir_full_pure(x) = target_full(x, args_full...)[3]
                    ep_full(x) = target_full(x, args_full...)[4]

                    optf_full = Optimization.OptimizationFunction(targetfun_full)
                    prob_full = Optimization.OptimizationProblem(optf_full, 
                        xinit, lb=llb1, ub=lub1, maxiters=maxiters_full)
                    sol_full = solve(prob_full, opt_alg)

                    # RESULTS
                    results_full[keyb][keyg][keyw] = vcat(sol_full.u, 0, 0, 0, 0, 
                        ep_full(sol_full.u), beta, gamma, w)

                    for i in 1:N
                        irvals_mix[i] = ir_full_mix(sol_full.u)
                        irvals_pure[i] = ir_full_pure(sol_full.u)
                    end

                    results_full[keyb][keyg][keyw][end-7] = Statistics.mean(irvals_mix)
                    results_full[keyb][keyg][keyw][end-6] = Statistics.mean(irvals_pure)
                    results_full[keyb][keyg][keyw][end-5] = Statistics.std(irvals_mix)
                    results_full[keyb][keyg][keyw][end-4] = Statistics.std(irvals_pure)
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
        params = Dict{String, Any}()
        params["ncpus"] = ncpus
        params["nparams"] = nparams
        params["betas"] = betas
        params["gammas"] = gammas
        params["wvals"] = wvals
        params["n"] = n
        params["N"] = N
        params["maxiters_acc"] = maxiters_acc
        params["maxiters_full"] = maxiters_full

        if whichf == "full" || whichf == "both"
            params["ncalls_full"] = COUNTERS["target_full"]
        end

        if whichf == "acc" || whichf == "both"
            params["ncalls_acc"] = COUNTERS["target_acc"]
        end

        paramfile = string(outpath, "params.json")
            
        open(paramfile, "w") do f
            JSON.print(f, params, 4)
        end
    end
end