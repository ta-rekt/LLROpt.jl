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
circuit_2bs = circuit_specs[circuitId]
edges = circuit_2bs["edges"]
input_edges = circuit_2bs["input_edges"]
nstates = circuit_2bs["nstates"]
off_ind = circuit_2bs["off_indicator"]
off_ind = BitArray(off_ind)
start_state = circuit_2bs["start_state"]
hitting_set = circuit_2bs["hitting_set"]
cr = 1

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
maxiters_full = 4000
whichf = "full"
fixed_conc = true   # is concentration fixed, or are we sweeping over it?
monte_carlo = false   # are we sampling trajectories and taking averages?

### Temperature parameter
# betasn = map(x->-tan(x), range(0*angle_incr, stop=0.5*angle_incr, length=16))
betas = map(x->tan(x), range(0, stop=pi/2 - 0.05, length=64))
# betas = [0.05]
nparams = length(betas)

### Other parameters
# gammas = [1.5,2.5,10]
# wvals = [0.05,0.2,1,5,10]
gammas = [2,8]
wvals = [0.1, 0.3, 1, 3, 10]

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
    edges1, edges2, iedges, cr, dc, on_ind1, on_ind2, beta, gamma, w = args
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

    pi_ = P_ss(d0)
    fir = FisherRate(d0, dx1, dx2)
    cov = CovarianceMatrix(d0, dx1, dx2, dy1, dy2)
    logSpec = log2((pi_[3]+pi_[7])/(pi_[3]+pi_[4]+pi_[5]+pi_[7]))

    (- 1/cov[1,1] - beta*logSpec, 1/cov[1,1], fir, logSpec)
end