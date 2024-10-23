@time begin
    using LLROpt
    using LinearAlgebra
    using ForwardDiff
    using BenchmarkTools
    using Distributions
    using StatsBase
    using Random
    using Base.Threads
    using Statistics
    using JSON
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: limit of IR_full as return rates -> infty 

@time let

    kvals = 10 .^ range(-1, 4, 20)
    ir_full_vals = Dict{Int64, Float32}()
    ir_acc_vals = Dict{Int64, Float32}()

    for (key, K) in enumerate(kvals)

        kboff = 0.0888
        konu = 1e-5
        kbon = 1e-5
        koffu = K
        kuoff = 1e-5
        konb = K
        kuon = K
        koffb = 1e-5

        m = LLROptModel(;
            weighted_edges = [(1,3,kboff), 
                              (1,2,konu), 
                              (2,4,kbon), 
                              (2,1,koffu), 
                              (3,1,kuoff), 
                              (3,4,konb), 
                              (4,2,kuon), 
                              (4,3,koffb)],
            input_edges = [(1,3), (2,4)],
            on_ind = BitArray([0,1,0,1]),
            c = 5.6,
            dc = 0.1,
            n = 1000000,
            idist = ones(4) ./ 4,
        )
        init!(m)
        ir_full!(m)
        irmax = 0.5*(m.dc/m.c)^2*(m.c*kboff*kuon)/(m.c*kboff+kuon)

        ir_full_vals[key] = m.llr[end] / (sum(m.rt)*irmax)
        ir_acc_vals[key] = 3*ir_acc(m.weighted_edges, m.input_edges, m.c, m.dc, [0,1,0,1])/(2*irmax)
    end

    file1 = "data/ir_full_scaling_output_20240726.json"
    file2 = "data/ir_acc_scaling_output_20240726.json"

    open(file1, "w") do f
        JSON.print(f, ir_full_vals, 4)
    end

    open(file2, "w") do f
        JSON.print(f, ir_acc_vals, 4)
    end
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: limits of DR for handcrafted circuit

@time let

    outdir = "20241022/dr_limits"
    fname = "dr_max_full_noac.json"
    acc = false

    if acc
        include("../spec/1bs_specificity.jl")
    else
        include("../spec/1bs_specificity_noac.jl")
    end

    outpath = "data/$(outdir)/"
    Base.Filesystem.mkpath(outpath)

    beta = 0
    gammas = [1.5, 2.5, 10]
    wvals = [2,5,100]
    N = 300
    keyb = 1

    # ACC = 1
    if acc
        # max ir
        # kboff = log10(0.0888)
        # konu = -4.95
        # kbon = -4.95
        # koffu = log10(5.0)
        # kuoff = -4.95
        # konb = log10(5.0)
        # kuon = log10(5.0)
        # koffb = -4.95

        # # max dr
        kboff = log10(0.0888)
        konu = -4.95
        kbon = log10(0.0888)
        koffu = -4.95
        kuoff = log10(5.0)
        konb = log10(5.0)
        kuon = log10(5.0)
        koffb = log10(5.0)

    # ACC = 0
    else
        kb = log10(0.0888)
        ku = log10(5.0)
    end

    results_full = Dict{Int64, Dict{Int64, Dict{Int64, Vector{Float32}}}}()
    results_full[keyb] = Dict{Int64, Dict{Int64, Vector{Float32}}}()

    for (keyg, gamma) in enumerate(gammas)
        results_full[keyb][keyg] = Dict{Int64, Vector{Float32}}()

        for (keyw, w) in enumerate(wvals)

            if acc
                logrates = kboff, konu, kbon, koffu, kuoff, konb, kuon, koffb
                logcw = log10(cr) + log10(w)
                logrates2 = [kuoff+log10(gamma), kboff+logcw, kbon+logcw, kuon+log10(gamma), 
                    konu, kboff, kbon, koffu, kuoff, konb, kuon, koffb, konb, koffb]
            else
                logrates = kb, ku
                logcw = log10(cr) + log10(w)
                logrates2 = [kb, ku, kb + logcw, ku + log10(gamma)]
            end

            we1 = [(u,v,r) for ((u,v),r) in zip(edges1, logrates)]
            we2 = [(u,v,r) for ((u,v),r) in zip(edges2, logrates2)]

            irvals_mix = Vector{Float32}(undef, N)
            irvals_pure = Vector{Float32}(undef, N)

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

            args_full = m1, m2, beta, gamma, w, true

            ir_full_mix(x) = target_full(x, args_full...)[2]
            ir_full_pure(x) = target_full(x, args_full...)[3]
            ep_full(x) = target_full(x, args_full...)[4]

            for i in 1:N
                irvals_mix[i] = ir_full_mix(logrates)
                irvals_pure[i] = ir_full_pure(logrates)
            end

            results_full[keyb][keyg][keyw] = vcat([i for i in logrates], 
                                                  Statistics.mean(irvals_mix), 
                                                  Statistics.mean(irvals_pure), 
                                                  Statistics.std(irvals_mix), 
                                                  Statistics.std(irvals_pure), 
                                                  ep_full(logrates), beta, gamma, w)

        end
    end

    file = string(outpath, fname)
                    
    open(file, "w") do f
        JSON.print(f, results_full, 4)
    end
end