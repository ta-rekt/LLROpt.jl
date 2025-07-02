using Optimization
using OptimizationBBO
using OptimizationCMAEvolutionStrategy
using OptimizationEvolutionary
using OptimizationManopt
using Distributions
using Cubature

### Biophysical parameters
a = 3e-3        # um
D_tf = 7.4      # um^2/s
D_cplx = 0.1    # um^2/s
dl_tf = 4*a*D_tf
ub = 5
lb = 1e-4

### Upper bounds for mixed ligands circuit
#   kboff, konb, kuon, kbon, koffu, kuoff
llb1 = ones(6) .* log10(lb)
lub1 = ones(6) .* log10(ub)
lub1[1] = log10(dl_tf)
lub1[4] = log10(dl_tf)

## load circuit specifications
circuits_file = "/home/tohme/.julia/dev/LLROpt/spec/circuits.json"
global circuit_specs = Dict()

open(circuits_file, "r") do f
    global circuit_specs = JSON.parse(f)
end

circuitId = "1bs_decoy_no_konu_new"
circuit = circuit_specs[circuitId]
edges = circuit["edges"]
input_edges = circuit["input_edges"]
nstates = circuit["nstates"]
off_ind = circuit["off_indicator"]
off_ind = BitArray(off_ind)
start_state = circuit["start_state"]
hitting_set = circuit["hitting_set"]
cr = 1
# beta = cr

maxevals = 2000
reltol = 1e-10
eps = 1e-15
N = 1   # legacy parameter, number of samples once optima have been found
n = 1

# initialization
xinit1 = (lub1 .+ llb1) ./ 2

### Initialization from stored data
# inpath = "/home/tohme/.julia/dev/LLROpt/data/20241118/dr-epsilon-hernan/bigsweep-cmaes-A-pn-smallw/ir_full_mix_vs_pure_pareto_pid"
inpath = ""

global d0 = OneBurstDecoyModel(;
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

dx1 = deepcopy(d0)
dx2 = deepcopy(d0)
dy1 = deepcopy(d0)
dy2 = deepcopy(d0)
dlist = [d0, dx1, dx2, dy1, dy2]

### Optimization parameters
opt_alg = BBO_adaptive_de_rand_1_bin_radiuslimited()
# opt_alg = Evolutionary.CMAES(μ = 5, λ = 10, sigma0 = 1)

maxiters_acc = 30000
maxiters_full = 8000
whichf = "both"
fixed_conc = true   # is concentration fixed, or are we sweeping over it?
monte_carlo = false   # are we sampling trajectories and taking averages?

### Temperature parameter
# betas = 10 .^ range(-2, stop=2, length=16)
betas = range(0, stop=2, length=16)
# betas = [0]
nparams = length(betas)

### Other parameters
gammas = [10]
wvals = [1e-5, 0.1, 10]
# gammas = [10]
# wvals = [0.01]

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

# returns - IR_acc_mix - beta * IR_acc_pure - gamma * EP
@counted function target_acc(logrates, args...)
    dlist, beta, gamma, w = args
    d0, __, __, _, _ = dlist

    d0.cw = d0.cr*w
    d0.gamma = gamma
    d0.logrates = logrates

    initAcc!(d0)

    pi_ = P_ss(d0)
    fir = FisherRateAcc(d0)
    logSpec = log2((pi_[3]+pi_[7])/(pi_[3]+pi_[4]+pi_[5]+pi_[7]))

    (- fir, 0, 0, logSpec)
end


@counted function target_full(logrates, args...)
    dlist, beta, gamma, w = args
    d0, dx1, dx2, dy1, dy2 = dlist

    for di in dlist
        di.logrates = logrates
        di.cr = d0.cr
        di.cw = d0.cr*w
        di.gamma = gamma
    end

    hx = 0.05 * d0.cr
    hy = 0.05 * d0.cw
    dx1.cr = d0.cr - hx
    dx2.cr = d0.cr - 2*hx
    dy1.cw = d0.cw - hy
    dy2.cw = d0.cw - 2*hy

    for di in dlist
        initFull!(di)
    end

    tauOffVal = tauOff(d0)
    tauOnVal = tauOn(d0)
    tau = tauOnVal + tauOffVal

    pi_ = P_ss(d0)
    fir = FisherRate(d0, dx1, dx2)
    cov = CovarianceMatrix(d0, dx1, dx2, dy1, dy2)
    # covOn = CovarianceOnMatrix(d0, dx1, dx2, dy1, dy2)
    # covOff = CovarianceOffMatrix(d0, dx1, dx2, dy1, dy2)
    logSpec = log2((pi_[3]+pi_[7])/(pi_[3]+pi_[4]+pi_[5]+pi_[7]))

    (- 1/cov[1,1], 1/covOff[1,1], 1/covOn[1,1], logSpec)
    # (- 1/(cov[1,1]*tau^(beta-1)), 0, 0, logSpec)
end
