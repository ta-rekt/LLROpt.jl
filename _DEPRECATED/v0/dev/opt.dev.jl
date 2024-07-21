## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# Optimization
let 
    global p = [1.0, 100.0]
    global rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
    x0 = zeros(2)
    f = OptimizationFunction(rosenbrock)
    lb = [-1.0, -1.0]
    ub = [1.0, 1.0]
    prob = Optimization.OptimizationProblem(f, x0, p; lb, ub)
    global sol = solve(prob, GCMAESOpt())
end