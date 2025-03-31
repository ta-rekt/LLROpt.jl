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
lb = 1e-5

### Upper bounds for mixed ligands circuit
#   kboff,   konb,    kuon,   kbon,   koffu,   kuoff,   konu
llb1 = ones(7) .* log10(lb)
lub1 = ones(7) .* log10(ub)
lub1[1] = log10(dl_tf)
lub1[4] = log10(dl_tf)

## circuit description
nstates = 6
edges = [(1, 2), (2, 1), (1, 6), (6, 1), (4, 3), (3, 4), 
         (4, 5), (5, 4), (6, 5), (4, 1), (2, 3), (1, 4)]
input_edges = [(1,2), (4,3)]
off_ind = BitArray([1,1,0,0,0,1])
# cr = 50
# dc = cr / 100
# beta = cr
# cw = 1e-3
gamma = 1 
maxevals = 10000
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
    cr = nothing,
    cw = nothing,
    gamma = nothing,
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
    cr = nothing,
    cw = nothing,
    gamma = nothing,
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
maxiters_full = 1500
whichf = "full"

### Temperature parameter
# angle_incr = 3.141592 / 1e3
# betasn = map(x->-tan(x), range(0*angle_incr, stop=0.5*angle_incr, length=128))
# betasp = map(x->tan(x), range(0*angle_incr, stop=1*angle_incr, length=128))
# betas = vcat(betasn, betasp)
# betas = 10 .^ range(-2, stop=4, length=16)
betas = [50]
nparams = length(betas)

### Other parameters
gammas = [10]
wvals = [1e-3]


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
    m1, m2, cr, gamma, cw = args

    m1.cr = cr
    m1.cw = cw
    m1.gamma = gamma
    m1.logrates = logrates

    m2.cr = 1.01*cr
    m2.cw = cw
    m2.gamma = gamma
    m2.logrates = logrates

    initOneBurst!(m1)
    initOneBurst!(m2)

    # print(logrates)
    EP = EP_ss(m1.Q)
    Jplus = 0
    pi_ = P_ss(m1.Q)
    # J = pi_[1] * m1.Q[1,2] - pi_[2] * m1.Q[2,1]
    # Jplus = J + abs(J)   # penalty for counterclockwise flux

    FisherInfoRate = FisherRate(m1, m2)
    logSpecificity = log2((cw/(cr*gamma))*(pi_[3])/(pi_[5]))

    (- FisherInfoRate + Jplus, logSpecificity, FisherInfoRate, EP)
    # return m1
end

