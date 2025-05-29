using Optimization
using OptimizationBBO
using OptimizationCMAEvolutionStrategy
using OptimizationEvolutionary
using OptimizationManopt
using Distributions
using Cubature
using JSON

### Biophysical parameters
a = 3e-3        # um
D_tf = 7.4      # um^2/s
D_cplx = 0.1    # um^2/s
dl_tf = 4*a*D_tf
ub = 5
lb = 1e-4

### Upper bounds for mixed ligands circuit
# "c*kboff","kuoff","konb","koffb","kuon","c*kbon","koffu","konu"
llb1 = ones(8) .* log10(lb)
lub1 = ones(8) .* log10(ub)
lub1[1] = log10(dl_tf)
lub1[6] = log10(dl_tf)


## load circuit specifications
circuits_file = "/home/tohme/.julia/dev/LLROpt/spec/circuits.json"
global circuit_specs = Dict()

open(circuits_file, "r") do f
    global circuit_specs = JSON.parse(f)
end

circuitId = "1bs"
circuit_1bs = circuit_specs[circuitId]
edges = circuit_1bs["edges"]
input_edges = circuit_1bs["input_edges"]
nstates = circuit_1bs["nstates"]
off_ind = circuit_1bs["off_indicator"]
off_ind = BitArray(off_ind)
start_state = circuit_1bs["start_state"]
hitting_set = circuit_1bs["hitting_set"]

cr = 40

maxevals = 20000
reltol = 1e-10
eps = 1e-15

# initialization
xinit1 = (lub1 .+ llb1) ./ 2

### Initialization from stored data
# inpath = "/home/tohme/.julia/dev/LLROpt/data/20241118/dr-epsilon-hernan/bigsweep-cmaes-A-pn-smallw/ir_full_mix_vs_pure_pareto_pid"
inpath = ""

global d = OneBurstDecoyModel(;
    circuitId = circuitId,
    nstates = nstates,
    edges = edges,
    input_edges = input_edges,
    logrates = nothing,
    cr = cr,
    cw = nothing,
    gamma = nothing,
    ind_set = off_ind,
    start_state = start_state,
    hitting_set = hitting_set,
    eps = eps,
    reltol = reltol,
    maxevals = maxevals,
)

d1 = deepcopy(d)
d2 = deepcopy(d)
dlist = [d, d1, d2]

### Optimization parameters
opt_alg = BBO_adaptive_de_rand_1_bin_radiuslimited()
# opt_alg = Evolutionary.CMAES(μ = 5, λ = 10, sigma0 = 1)

maxiters_acc = 30000
maxiters_full = 500
whichf = "acc"
fixed_conc = false   # is concentration fixed, or are we sweeping over it?
monte_carlo = false
N = 1  # legacy parameter, number of repeats once optima have been found
n = 1

### Temperature parameter
betas = vcat(range(6e-2, stop=1e-3, length=128), 
             10 .^ range(-3, stop=-7, length=32))
# betas = [0.1]
nparams = length(betas)

### Other parameters
gammas = [0]
wvals = [0]


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

# IR ACC
@counted function target_acc(logrates, args...)
    dlist, beta, gamma, w = args
    d, d1, d2 = dlist

    d.logrates = logrates

    initAcc!(d)

    EP = EP_ss(d)
    fir = FisherRateAcc(d)

    (- fir + beta*EP, fir, 0, EP)
end

# IR FULL
@counted function target_full(logrates, args...)
    dlist, beta, gamma, w = args
    d, d1, d2 = dlist
    
    h = 0.01 * d.cr
    d1.cr = cr - h
    d2.cr = cr - 2*h
    d.logrates = logrates
    d1.logrates = logrates
    d2.logrates = logrates

    initFull!(d)
    initFull!(d1)
    initFull!(d2)

    EP = EP_ss(d)

    fir = FisherRate(d, d1, d2)
    firOff = FisherRateOff(d, d1, d2)

    (- fir + beta*EP, fir, firOff, EP)
end
