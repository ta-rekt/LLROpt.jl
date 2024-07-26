## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
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
# DOING: testing ir_full
@time let
    # Random.seed!(123)

    global m0 = LLROptModel(;
        weighted_edges = [(1,3,8.87533829e-02), 
                          (1,2,1.44048849e-04), 
                          (2,4,8.46399804e-04), 
                          (2,1,2.73183156e+00), 
                          (3,1,2.27367035e-05), 
                          (3,4,4.99957560e+00), 
                          (4,2,4.72051807e+00), 
                          (4,3,1.59700801e-03)],
        input_edges = [(1,3), (2,4)],
        on_ind = BitArray([0,1,0,1]),
        c = 5.6,
        dc = 0.56,
        n = 1000000,
        idist = ones(4) ./ 4,
    )
    init!(m0)
    # models = [deepcopy(m0) for t in 1:nthreads()]
    # lk = ReentrantLock()
    # N = 1000
    acc = 0
    # @threads :static for it in 1:N
    # for it in 1:N
    # t = threadid()
    # m = models[t]
    
    ir_full!(m0)

    # lock(lk) do
    acc += m0.llr[end] / (sum(m0.rt))
    # end

    # end
    # @show acc/N
    @show acc
    # N == 100 && @assert acc == 0.20647115491313528
    # N == 1000 && @assert acc == 1.7704159503081027
    nothing
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing text IO

let 
    file = "test.txt"

    open(file, "w") do f
        for i in 1:5
            line = string(i, i+1)
            
            println(f, line)
        end
    end
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing JSON
using JSON

let 
    Random.seed!(123)
    nrow = 3
    ncol = 2
    data = rand(nrow,ncol)
    dict = Dict(i => data[i:nrow:nrow*ncol] for i = 1:nrow)
    
    file = "test.json"

    open(file, "w") do f
        JSON.print(f, dict, 4)

    end
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing command-line input

let
    s = readline(stdin)
end