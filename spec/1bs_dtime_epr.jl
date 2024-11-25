using Optimization
using OptimizationBBO
using OptimizationCMAEvolutionStrategy
using OptimizationEvolutionary
using OptimizationManopt
using Distributions

### Biophysical parameters
a = 3e-3        # um
D_tf = 7.4      # um^2/s
D_cplx = 0.1    # um^2/s
c = 5.6        # um^-3
dc = 0.1 * c
dl_tf = 4*a*D_tf
ub = 5
lb = 1e-5
max_ir_full = 0.5 * (dc/c)^2 * (c*dl_tf*ub)/(c*dl_tf + ub)

### Upper bounds for pure ligand circuit
llb = ones(8) .* log10(lb)
lub = ones(8) .* log10(ub)
lub[1] = log10(dl_tf)
lub[3] = log10(dl_tf)


### Circuit desciptions
edges = [(1,3), (1,2), (2,4), (2,1), (3,1), (3,4), (4,2), (4,3)]
input_edges = [(1,3), (2,4)]
on_ind = BitArray([0,1,0,1])
idist = ones(4) ./ 4

### Initialization from stored data
# inpath = "/home/tohme/.julia/dev/LLROpt/data/20241125/ir-dr-hernan/coarse-sweep-1-alg=de-start=MGR-lg=global-whichf=simfull-maxiters=3000-sigma=3e-5-N=3000-nparams=1/ir_full_mix_vs_pure_pareto_pid"
inpath = ""
# inpath = "prevbeta"

### Initialization at midpoint of bounds (trivial)
# xinit1 = (lub .+ llb) ./ 2

### Local search around MGR circuit
xinit1 = [log10(0.0888)- 0.01,
          -5 + 0.01,
          log10(0.0888)- 0.01,
          log10(5)- 0.01,
          -5 + 0.01,
          log10(5)- 0.01,
          log10(5) - 0.01,
          -5 + 0.01]

we = [(u,v,r) for ((u,v),r) in zip(edges, xinit1)]

### Optimization parameters
opt_alg = BBO_adaptive_de_rand_1_bin_radiuslimited()
# opt_alg = Evolutionary.CMAES(μ = 5, λ = 10, sigma0 = 0.01)

maxiters_acc = 10000
maxiters_full = 1000
n = 5000
N = 1
k = 100
whichf = "full"

### Temperature parameter
angle_inc = 3.141592/2 - 0.01
# betasn = map(x->-tan(x), range(0*angle_inc, stop=0.5*angle_inc, length=64))
betas = map(x->tan(x), range(0*angle_inc, stop=1*angle_inc, length=16))
# betas = vcat(betasn, betasp)
nparams = length(betas)

### Other parameters
e = 0.25

### objective functions

const COUNTERS = Dict{String, Int}()

macro counted(f)
    name = f.args[1].args[1]
    name_str = String(name)
    body = f.args[2]
    counter_code = quote
        if !haskey(COUNTERS, $name_str)
            COUNTERS[$name_str] = 0
        end
        COUNTERS[$name_str] += 1
    end
    insert!(body.args, 1, counter_code)
    return f
end

# returns - IR_full_mix - beta * IR_full_pure - gamma * EP
@counted function ir_full_log(logrates, args...)
    m, beta, e = args

    for (keyb, val) in enumerate(logrates)
        (u,v,r) = m.weighted_edges[keyb]
        m.weighted_edges[keyb] = (u,v,10 ^ val)
    end

    init!(m)
    EP = EP_ss(m.Q_c)

    Jplus = 0
    pi = P_ss(m.Q_c)
    J = pi[1] * m.Q_c[1,2] - pi[2] * m.Q_c[2,1]
    Jplus = J + abs(J)   # penalty for counterclockwise flux

    K = log((1-e)/e)
    stop_times = Vector{Float64}(undef, m.k)

    for i in 1:m.k
        ir_full!(m)
        time = cumsum(m.rt)
        q = findfirst(x -> x > K || x < (-K), m.llr[2:end])
        if q === nothing
            stop_times[i] = time[end]
        else
            stop_times[i] = time[q]
        end
    end

    # mean decision time
    mean_dt = Statistics.mean(stop_times)
    # var_dt = Statistics.var(stop_times)

    (mean_dt - beta*EP + Jplus, mean_dt, EP)

end
