module LLROpt

    using LinearAlgebra
    using ForwardDiff
    using StatsBase
    using Distributions
    using ExtractMacro


    include("0.types.jl")
    include("ctmc.jl")
    include("ir_full.jl")
    include("ll_full.jl")

end
