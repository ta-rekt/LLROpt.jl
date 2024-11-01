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
# DOING: inferring circuits from their traces (2 state)

function ll(x, args)

end

@time let

    include("../spec/1bs_specificity_noac.jl")

    kb = log10(0.0888)
    ku = log10(5.0)
    logrates = kb, ku   # true circuit
    lub = 5
    llb = -5
    n = 1000

    we1 = [(u,v,r) for ((u,v),r) in zip(edges1, logrates)]
    m0 = LLROptModel(;
        weighted_edges = we1,
        input_edges = input_edges,
        on_ind = on_ind1,
        c = cr,
        dc = dc,
        n = n,
        k = k,
        idist = idist1,
    )

    init!(m0)
    generate!(m0)
    get_residence_time!(m0)
    
    m0.ll_c = ll_full(m0.Q_c, m0.ind_set, m0.t_set, m0.t_comp, m0.idist_set)
    score = m0.ll_c / n

    edges_fc = vec([(i,j) for i in [1,2,3,4], j in [1,2,3,4]])
    on_ind_fc = BitArray([0,1,0,1])
    idist_fc = ones(4) ./ 4
    x_init = zeros(16)
    we_fc = [(u,v,r) for ((u,v),r) in zip(edges_fc, x_init)]

    m = LLROptModel(;
        weighted_edges = we_fc,
        input_edges = input_edges,
        on_ind = on_ind_fc,
        idist = idist_fc,
        c = cr,
        dc = dc,
        n = n,
        k = k
    )

    args = edges_fc, m

end