using Optimization
using OptimizationBBO
using OptimizationCMAEvolutionStrategy
using OptimizationEvolutionary
using OptimizationManopt
using Distributions
using Cubature

### biophysical parameters
a = 3e-3        # ums
D_tf = 7.4      # um^2/s
D_cplx = 0.1    # um^2/s
c = 5.6         # um^-3
dc = 0.1 * c
dl_tf = 4*a*D_tf
max_ir_full = 0.5 * (dc/c)^2 * c*4*a*D_tf

### 6-binding site circuit parameters
edges = [(1 ,2 ), (2 ,1 ),
         (2 ,3 ), (3 ,2 ),
         (3 ,4 ), (4 ,3 ),
         (4 ,5 ), (5 ,4 ),
         (5 ,6 ), (6 ,5 ),
         (6 ,7 ), (7 ,6 )]

nstates = 7
input_edges = [(i,i+1) for i in range(1,6)]
off_ind = [1,1,1,0,0,0,0]
off_ind = BitArray(off_ind)
k = 3
hset = [k+1]
idist = ones(7) ./ 7
e = (-4,11)                   # symmetry-breaking edge
fixed_conc = false   # is concentration fixed, or are we sweeping over it?
monte_carlo = false

### Bounding box on rates
ub = 5
lb = 1e-4
llb1 = ones(12) .* log10(lb)

#    rate upper bounds |       edges
ubs = [6*dl_tf, 1*ub,   # (1 ,2 ), (2 ,1 )
       5*dl_tf, 2*ub,   # (2 ,3 ), (3 ,2 )
       4*dl_tf, 3*ub,   # (3 ,4 ), (4 ,3 )
       3*dl_tf, 4*ub,   # (4 ,5 ), (5 ,4 )
       2*dl_tf, 5*ub,   # (5 ,6 ), (6 ,5 )
       1*dl_tf, 6*ub]   # (6 ,7 ), (7 ,6 )

lub1 = log10.(ubs)

### Initialization from stored data
# inpath = "/home/tohme/.julia/dev/LLROpt/data/20241118/dr-epsilon-hernan/bigsweep-cmaes-A-pn-smallw/ir_full_mix_vs_pure_pareto_pid"
inpath = ""

### Optimization parameters
opt_alg = BBO_adaptive_de_rand_1_bin_radiuslimited()
# opt_alg = Evolutionary.CMAES(μ = 5, λ = 10, sigma0 = 1)

maxevals = 50000
reltol = 1e-12
eps = 1e-30

global m1 = OneBurstDecoyModel(;
    nstates = nstates,
    edges = edges,
    input_edges = input_edges,
    logrates = nothing,
    cr = nothing,
    cw = nothing,
    gamma = nothing,
    ind_set = off_ind,
    start_state = k,
    hitting_set = hset,
    eps = eps,
    reltol = reltol,
    maxevals = maxevals,
)

global m2 = OneBurstDecoyModel(;
    nstates = nstates,
    edges = edges,
    input_edges = input_edges,
    logrates = nothing,
    cr = nothing,
    cw = nothing,
    gamma = nothing,
    ind_set = off_ind,
    start_state = k,
    hitting_set = hset,
    eps = eps,
    reltol = reltol,
    maxevals = maxevals,
)

### Optimization parameters
maxiters_acc = 10000
maxiters_full = 3000
sigma0 = log10(ub) / 4
xinit1 = (lub1 .+ llb1) ./ 2
we = [(u,v,r) for ((u,v),r) in zip(edges, xinit1)]
n = 1000000
k = 1
N = 2000
whichf = "full"
fixed_conc = false

### temperature parameter
# betas = vcat(range(1e-1, stop=1e-5, length=16), 10 .^ range(-4.8, stop=-9, length=16))
betas = 10 .^ range(-1, 2, length=16)
# betas = [1]
nparams = length(betas)
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

# FIR ACC constraining sharpness
@counted function target_full(logrates, args...)
    m1, m2, beta, gamma, cw = args
    
    cr = beta
    dc = cr/20
    m1.cr = cr
    m2.cr = cr + dc
    m1.logrates = logrates
    m2.logrates = logrates

    init6bs!(m1)
    init6bs!(m2)
    
    # clockwise bias by forcing flow along edge e
    Jplus = 0
    m, n = e
    pi_ = P_ss(m1)
    pion = pi_on(m1)

    if m >= 0
        J = pi_[m] * m1.Q[m,n] - pi_[n] * m1.Q[n,m]
        Jplus = J + abs(J)
    end

    FIR = FisherRate(m1, m2)
    EP = EP_ss(m1)
    s = abs(sharpness(m1, m2))
    S = s / (pion * (1 - pion))

    # (- FIR +(S-5)^2, cr, FIR, S)
    (- FIR, cr, FIR, S)
end

# IR FULL
function ir_full_log(logrates, args...)
    m, beta, use_mean, e = args
end