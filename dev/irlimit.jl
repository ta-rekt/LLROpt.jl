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

    kvals = range(-1, 4, 20)
    ir_full_vals_1bs = Dict{Int64, Dict{String, Float32}}()
    ir_acc_vals_1bs = Dict{Int64, Dict{String, Float32}}()
    ir_full_vals_2bs = Dict{Int64, Dict{String, Float32}}()
    ir_acc_vals_2bs = Dict{Int64, Dict{String, Float32}}()

    for (key, K) in enumerate(kvals)

        # 1bs max-V circuit, full (merry-go-around circuit)
        kboff = -1.0613204
        konu = -5.0
        kbon = -5.0
        koffu = K
        kuoff = -5.0
        konb = K
        kuon = K
        koffb = -5.0

        # 1bs max-V circuit, mRNA
        # kboff = 0.0888
        # konu = 1e-5
        # kbon = 1e-5
        # koffu = K
        # kuoff = 1e-5
        # konb = K
        # kuon = K
        # koffb = 1e-5

        logrates1 = [kboff, konu, kbon, koffu, kuoff, konb, kuon, koffb]

        # 2bs max-V circuit, full
        kboff = -1.0613204
        konu = -2.98726
        kbon = -1.0868188
        koffu = K
        kuoff = -2.276372
        kon1b = K
        kuon = K
        koff1b = -2.2888334 
        kon2b = K
        koff2b = K

        # 2bs max-V circuit, mRNA
        # kboff = -1.0515872
        # konu =  -2.999998
        # kbon =  -1.0515875
        # koffu = 0.69897
        # kuoff =  -2.999923
        # kon1b = 0.69897
        # kuon = 0.69896984
        # koff1b = -2.9996622  
        # kon2b = 0.69897
        # koff2b = -2.9994574

        logrates2 = [kboff+log10(2), kuoff,        kuon,  kbon+log10(2), kon1b, kboff, kbon, 
                koff1b,       kuoff+log10(2), kon2b, kuon+log10(2), koff2b, konu,  koffu]

        edges1 = [(1,3), (1,2), (2,4), (2,1), (3,1), (3,4), (4,2), (4,3)]
        input_edges1 = [(1,3), (2,4)]
        on_ind1 = BitArray([0,1,0,1])
        idist1 = ones(4) ./ 4

        edges2 = [(5,1), (1,5), (2,6), (6,2), (1,2), (1,3), (2,4),
                (2,1), (3,1), (3,4), (4,2), (4,3), (5,6), (6,5)]
        input_edges2 = [(1,3), (2,4), (5,1), (6,2)]
        on_ind2 = BitArray([0,1,0,1,0,1])
        idist2 = ones(6) ./ 6

        we1 = [(u,v,10.0^r) for ((u,v),r) in zip(edges1, logrates1)]
        we2 = [(u,v,10.0^r) for ((u,v),r) in zip(edges2, logrates2)]

        n = 10000000

        global m1 = LLROptModel(;
            weighted_edges = we1,
            input_edges = input_edges1,
            on_ind = on_ind1,
            c = 5.6,
            dc = 0.1,
            n = n,
            k = 1,
            idist = idist1,
        )

        global m2 = LLROptModel(;
            weighted_edges = we2,
            input_edges = input_edges2,
            on_ind = on_ind2,
            c = 5.6,
            dc = 0.1,
            n = n,
            k = 1,
            idist = idist2,
        )

        init!(m1)
        init!(m2)
        ir_full!(m1)
        ir_full!(m2)

        ir_full_vals_1bs[key] = Dict("k" => K, "val" => m1.llr[end] / sum(m1.rt))
        # ir_acc_vals_1bs[key] = ir_acc(m1.weighted_edges, 
        #                               m1.input_edges, 
        #                               m1.c, 
        #                               m1.dc, 
        #                               m1.on_ind)

        ir_full_vals_2bs[key] = Dict("k" => K, "val" => m2.llr[end] / sum(m2.rt))
        # ir_acc_vals_2bs[key] = ir_acc(m2.weighted_edges, 
        #                             m2.input_edges, 
        #                             m2.c, 
        #                             m2.dc, 
        #                             m2.on_ind)
    end

    date = 20241211
    outdir = "$(date)/irlimit/"
    outpath = "data/$(outdir)/"
    Base.Filesystem.mkpath(outpath)

    file1 = string(outpath, "ir_full_1bs_scaling.json")
    file2 = string(outpath, "ir_full_2bs_scaling.json")

    open(file1, "w") do f
        JSON.print(f, ir_full_vals_1bs, 4)
    end

    open(file2, "w") do f
        JSON.print(f, ir_full_vals_2bs, 4)
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