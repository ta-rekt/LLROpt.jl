@time begin
    using LLROpt
    using LinearAlgebra
    using ForwardDiff
    using BenchmarkTools
    using Distributions
    using StatsBase
    using Random
    using Base.Threads
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
