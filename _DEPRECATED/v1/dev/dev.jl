## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
@time begin
    using LLROpt
    using LinearAlgebra
    using ForwardDiff
    using BenchmarkTools
    using Distributions
    using StatsBase
    using Random
    using Optimization
    using OptimizationGCMAES
end


## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing ir_full

@time let 
    Random.seed!(123)

    weighted_edges = [(1, 2, 0.5), (2, 1, 2.0)]
    input_edges = [(1, 2)]
    on_ind = BitArray([0, 1])
    c = 1.0
    dc = c * 1e-1
    n = 1000
    idist = ones(2) ./ 2
    
    N = 1000
    acc = 0.0
    for it in 1:N
        onrt, offrt, llr = ir_full(weighted_edges, input_edges, c, dc, on_ind, idist, n)
        acc += llr[end] / (sum(onrt) + sum(offrt))
    end
    @show acc/N
end

