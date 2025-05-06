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
#   kboff,   konb,    kuon,   kbon,   koffu,   kuoff // konu
llb1 = ones(6) .* log10(lb)
lub1 = ones(6) .* log10(ub)
lub1[1] = log10(dl_tf)
lub1[4] = log10(dl_tf)

## circuit description
nstates = 6
edges = [(1, 2), (2, 1), (1, 6), (6, 1), (4, 3), (3, 4), 
         (4, 5), (5, 4), (6, 5), (4, 1), (2, 3)]
input_edges = [(1,2), (4,3)]
off_ind = BitArray([1,1,0,0,0,1])
cr = 1
# beta = cr

maxevals = 50000
reltol = 1e-10
eps = 1e-30
N = 1   # legacy parameter, number of samples once optima have been found
n = 1

# initialization
xinit1 = (lub1 .+ llb1) ./ 2

### Initialization from stored data
# inpath = "/home/tohme/.julia/dev/LLROpt/data/20241118/dr-epsilon-hernan/bigsweep-cmaes-A-pn-smallw/ir_full_mix_vs_pure_pareto_pid"
inpath = ""

m1 = OneBurstDecoyModel(;
    nstates = nstates,
    edges = edges,
    input_edges = input_edges,
    logrates = nothing,
    cr = cr,
    cw = nothing,
    gamma = nothing,
    start_state = 1,
    hitting_set = [3,4,5],
    ind_set = off_ind,
    eps = eps,
    reltol = reltol,
    maxevals = maxevals,
)

m2 = OneBurstDecoyModel(;
    nstates = nstates,
    edges = edges,
    input_edges = input_edges,
    logrates = nothing,
    cr = 1.05*cr,
    cw = nothing,
    gamma = nothing,
    start_state = 1,
    hitting_set = [3,4,5],
    ind_set = off_ind,
    eps = eps,
    reltol = reltol,
    maxevals = maxevals,
)

# initOneBurst!(m1)
# initOneBurst!(m2)

### Optimization parameters
opt_alg = BBO_adaptive_de_rand_1_bin_radiuslimited()
# opt_alg = Evolutionary.CMAES(μ = 5, λ = 10, sigma0 = 1)

maxiters_acc = 30000
maxiters_full = 3000
whichf = "full"
fixed_conc = true   # is concentration fixed, or are we sweeping over it?
monte_carlo = false   # are we sampling trajectories and taking averages?

### Temperature parameter
betas1 = map(x->tan(x), range(0, stop=pi/15, length=32))
betas2 = map(x->tan(x), range(pi/15, stop=pi/2 - 0.1, length=32))
betas = vcat(betas1, betas2)
# betas = 10 .^ range(-1, stop=3, length=32)
# betas = [0.05]
nparams = length(betas)

### Other parameters
gammas = [1.5,10]
wvals = [0.01,5]


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
    m1, m2, beta, gamma, w = args

    m1.cw = m1.cr*w
    m1.gamma = gamma
    m1.logrates = logrates

    m2.cw = m1.cr*w
    m2.gamma = gamma
    m2.logrates = logrates

    initDecoys!(m1)
    initDecoys!(m2)

    # print(logrates)
    EP = EP_ss(m1)
    Jplus = 0
    pi_ = P_ss(m1)
    # J = pi_[1] * m1.Q[1,2] - pi_[2] * m1.Q[2,1]
    # Jplus = J + abs(J)   # penalty for counterclockwise flux

    fir = FisherRate(m1, m2)
    firOff = FisherRateOff(m1, m2)
    firOn = FisherRateOn(m1, m2)
    logSpec = log2((pi_[3])/(pi_[3]+pi_[4]+pi_[5]))

    (- fir - beta*logSpec, fir, logSpec, EP)
end

