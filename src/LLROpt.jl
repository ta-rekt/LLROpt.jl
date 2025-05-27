module LLROpt

    using LinearAlgebra
    import LinearAlgebra: exp!
    using ForwardDiff
    using StatsBase
    using Distributions
    using ExtractMacro
    using Base.Threads
    using Cubature
    using StaticArrays


    include("0.types.jl")
    include("ctmc.jl")
    include("fpt.jl")
    include("fisher.jl")
    include("ir_full.jl")
    include("ll_full.jl")
    include("generate.jl")
    include("init.jl")
    include("get_residence_time.jl")

end
