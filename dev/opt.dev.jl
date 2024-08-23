begin
    using LLROpt
    using Optimization
end
##. - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# GCMAES minimal working example with parabolic function
# --> doesn't respect search range
let 
    using Optimization, OptimizationOptimJL

    rosenbrock(x, p) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    p = [1.0, 100.0]
    prob = Optimization.OptimizationProblem(rosenbrock, x0, p)
    sol = solve(prob, Optim.NelderMead())

end
##. - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# Optimization minimal working example with parabolic function
begin
    using OptimizationCMAEvolutionStrategy
    using OptimizationGCMAES
    using OptimizationBBO
    using OptimizationOptimJL
end

function f(x, p)
    (x[1] - p[1])^2 + (x[2] - p[2])^2
end 

let 
    x0 = [0.0,0.0]
    sigma0 = 0.5
    maxiters = 50000
    lb = [0.0,0.0]
    ub = [4.0,5.0]
    p = [-1,6]

    optf = Optimization.OptimizationFunction(f)
    prob = Optimization.OptimizationProblem(optf, x0, p, lb=lb, ub=ub, maxiters=maxiters)
    sol = solve(prob, Optim.ParticleSwarm())
    
end 
