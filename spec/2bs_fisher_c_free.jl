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
lb = 1e-3

### Upper bounds for mixed ligands circuit
#   kboff,   konb,    kuon,   kbon,   koffu,   kuoff //   konu
llb1 = ones(11) .* log10(lb)
lub1 = ones(11) .* log10(ub)
lub1[1] = log10(2*dl_tf)
lub1[4] = log10(2*dl_tf)
lub1[6] = log10(dl_tf)
lub1[7] = log10(dl_tf)
lub1[8] = log10(2*ub)
lub1[10] = log10(2*ub)


## load circuit specifications
circuits_file = "/home/tohme/.julia/dev/LLROpt/spec/circuits.json"
global circuit_specs = Dict()

open(circuits_file, "r") do f
    global circuit_specs = JSON.parse(f)
end

circuitId = "2bs_no_konu"
circuit_2bs = circuit_specs[circuitId]
edges = circuit_2bs["edges"]
input_edges = circuit_2bs["input_edges"]
nstates = circuit_2bs["nstates"]
off_ind = circuit_2bs["off_indicator"]
off_ind = BitArray(off_ind)
start_state = circuit_2bs["start_state"]
hitting_set = circuit_2bs["hitting_set"]

# cr = 100
# dc = cr / 10
# beta = cr
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
    cr = nothing,
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
maxiters_full = 2000
whichf = "full"
fixed_conc = false   # is concentration fixed, or are we sweeping over it?
monte_carlo = false
N = 1  # legacy parameter, number of repeats once optima have been found
n = 1

### Temperature parameter
# angle_incr = 3.141592 / 1e3
# betasn = map(x->-tan(x), range(0*angle_incr, stop=0.5*angle_incr, length=128))
# betasp = map(x->tan(x), range(0*angle_incr, stop=1*angle_incr, length=128))
# betas = vcat(betasn, betasp)
betas = 10 .^ range(-1, stop=2, length=16)
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
    edges, iedges, c, dc, on_ind, beta, e = args
    nrates = length(edges)
    kboff, konu, kbon, koffu, kuoff, kon1b, kuon, koff1b, kon2b, koff2b = logrates

    # edges = [(5,1),         (1,5),         (2,6),  (6,2),     (1,2),  (1,3),  (2,4),
    #          (2,1),         (3,1),         (3,4),  (4,2),     (4,3),  (5,6),  (6,5)]

    logrates2 = [kboff+log10(2), kuoff,        kuon,  kbon+log10(2), kon1b, kboff, kbon,
                 koff1b,       kuoff+log10(2), kon2b, kuon+log10(2), koff2b, konu,  koffu]

    rates = 10 .^ logrates2
    wedges = Vector{Tuple{Int64, Int64, Float32}}(undef, nrates)

    for i in 1:nrates
        w1 = edges[i][1]
        w2 = edges[i][2]
        w3 = rates[i]

        wedges[i] = (w1, w2, w3)
    end

    Q = build_mat(wedges, iedges, c)

    # clockwise bias by forcing flow along edge e
    Jplus = 0
    m, n = e
    if m >= 0
        pi = P_ss(Q)
        J = pi[m] * Q[m,n] - pi[n] * Q[n,m]
        Jplus = J + abs(J)
    end

    IR = ir_acc(wedges, iedges, beta, beta/10, on_ind)

    (- IR + Jplus, IR, beta)
end

# IR FULL
@counted function target_full(logrates, args...)
    dlist, beta, gamma, w = args
    d, d1, d2 = dlist
    
    cr = beta
    d.cr = cr
    h = 0.01 * d.cr
    d1.cr = cr - h
    d2.cr = cr - 2*h
    d.logrates = logrates
    d1.logrates = logrates
    d2.logrates = logrates

    initFull!(d)
    initFull!(d1)
    initFull!(d2)

    # print(logrates)
    EP = EP_ss(d)

    fir = FisherRate(d, d1, d2)
    firOff = FisherRateOff(d, d1, d2)
    firOn = FisherRateOn(d, d1, d2)

    (- fir, firOff, firOn, EP)
end
