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
    using Base.Threads
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
@time let
    Random.seed!(123)

    m = LLROptModel(;
        weighted_edges = [(1, 2, 0.5), (2, 1, 2.0)],
        input_edges = [(1, 2)],
        on_ind = BitArray([0, 1]),
        c = 1.0,
        dc = 1e-1,
        n = 1000,
        idist = ones(2) ./ 2,
    )
    init!(m)

    N = 10000
    acc = 0
    # @threads :static for it in 1:N
    for it in 1:N
        t = threadid()
        ir_full!(m)
        acc += m.llr[end] / (sum(m.onrt) + sum(m.offrt))
    end
    @show acc/N
    @show acc
    # N == 100 && @assert acc == 0.20647115491313528
    # N == 1000 && @assert acc == 1.7704159503081027
    nothing
end


## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing ir_full

@time let 
    Random.seed!(123)

    global weighted_edges = [(1, 2, 0.5), (2, 1, 2.0)]
    global input_edges = [(1, 2)]
    global on_ind = BitArray([0, 1])
    global c = 1.0
    global dc = c * 1e-1
    global n = 1000
    global idist = ones(2) ./ 2
    
    N = 1000
    acc = 0.0
    for it in 1:N
        # onrt, offrt, llr = ir_full(weighted_edges, input_edges, c, dc, on_ind, idist, n)
        ir_full(weighted_edges, input_edges, c, dc, on_ind, idist, n)
        # acc += llr[end] / (sum(onrt) + sum(offrt))
    end
    @show acc/N
    @show acc
    # @assert acc == 0.20647115491313528 # N = 100
    # @assert acc == 1.7704159503081027 # N = 1000
end

