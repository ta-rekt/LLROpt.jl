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
         (6 ,7 ), (7 ,6 ),
         (8 ,9 ), (9 ,8 ),
         (9 ,10), (10,9 ),
         (10,11), (11,10),
         (11,12), (12,11),
         (12,13), (13,12),
         (13,14), (14,13),
         (1 ,8 ), (8 ,1 ),
         (2 ,9 ), (9 ,2 ),
         (3 ,10), (10,3 ),
         (4 ,11), (11,4 ),
         (5 ,12), (12,5 ),
         (6 ,13), (13,6 ),
         (7 ,14), (14,7 )]

### julia-python converter
# input_labels = {(4,0):"(5,1)",
#                 (0,4):"(1,5)",
#                 (1,5):"(2,6)",
#                 (5,1):"(6,2)",
#                 (0,1):"(1,2)", 
#                 (0,2):"(1,3)", 
#                 (1,3):"(2,4)", 
#                 (1,0):"(2,1)", 
#                 (2,0):"(3,1)", 
#                 (2,3):"(3,4)", 
#                 (3,1):"(4,2)", 
#                 (3,2):"(4,3)",
#                 (4,5):"(5,6)",
#                 (5,4):"(6,5)"}

input_edges = [(i,i+1) for i in range(1,13) if i != 7]
on_ind = vcat(zeros(Bool, 7), ones(Bool, 7))
on_ind = BitArray(on_ind)
idist = ones(14) ./ 14
e = (-1,8)                   # symmetry-breaking edge

### Bounding box on rates
ub = 5
lb = 1e-3
llb = ones(38) .* log10(lb)
# lub = ones(38) .* log10(ub)

#    rate upper bounds |       edges
ubs = [6*dl_tf, 1*ub,   # (1 ,2 ), (2 ,1 )
       5*dl_tf, 2*ub,   # (2 ,3 ), (3 ,2 )
       4*dl_tf, 3*ub,   # (3 ,4 ), (4 ,3 )
       3*dl_tf, 4*ub,   # (4 ,5 ), (5 ,4 )
       2*dl_tf, 5*ub,   # (5 ,6 ), (6 ,5 )
       1*dl_tf, 6*ub,   # (6 ,7 ), (7 ,6 )
       6*dl_tf, 1*ub,   # (8 ,9 ), (9 ,8 )
       5*dl_tf, 2*ub,   # (9 ,10), (10,9 )
       4*dl_tf, 3*ub,   # (10,11), (11,10)
       3*dl_tf, 4*ub,   # (11,12), (12,11)
       2*dl_tf, 5*ub,   # (12,13), (13,12)
       1*dl_tf, 6*ub,   # (13,14), (14,13)
       ub,      ub,     # (1 ,8 ), (8 ,1 )
       ub,      ub,     # (2 ,9 ), (9 ,2 )
       ub,      ub,     # (3 ,10), (10,3 )
       ub,      ub,     # (4 ,11), (11,4 )
       ub,      ub,     # (5 ,12), (12,5 )
       ub,      ub,     # (6 ,13), (13,6 )
       ub,      ub]     # (7 ,14), (14,7 )

lub = log10.(ubs)

### Initialization from stored data
# inpath = "/home/tohme/.julia/dev/LLROpt/data/20241118/dr-epsilon-hernan/bigsweep-cmaes-A-pn-smallw/ir_full_mix_vs_pure_pareto_pid"
inpath = ""

global m1 = OneBurstDecoyModel(;
    nstates = nstates,
    edges = edges,
    input_edges = input_edges,
    logrates = nothing,
    cr = nothing,
    cw = nothing,
    gamma = nothing,
    ind_set = off_ind,
    start_state = 1,
    hitting_set = [8,9,10,11,12,13,14],
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
    start_state = 1,
    hitting_set = [8,9,10,11,12,13,14],
    eps = eps,
    reltol = reltol,
    maxevals = maxevals,
)

### Optimization parameters
maxiters_acc = 30000
maxiters_full = 8000
sigma0 = log10(ub) / 4
x0 = (lub .+ llb) ./ 2
we = [(u,v,r) for ((u,v),r) in zip(edges, x0)]
n = 1000000
k = 1
N = 2000
whichf = "acc"

### temperature parameter
# betas = vcat(range(3e-4, stop=1e-5, length=15), 10 .^ range(-4.8, stop=-9, length=15))
betas = range(1e-1, 1e2, length=16)
nparams = length(betas)

# IR ACC
function ir_acc_log(logrates, args...)
    m1, m2, cr, gamma, cw = args
    
    m1.cr = cr
    m2.cr = cr*1.1
    m1.logrates = logrates
    m2.logrates = logrates

    init6bs!(m1)
    init6bs!(m2)

    EP = EP_ss(m1)
    
    # clockwise bias by forcing flow along edge e
    Jplus = 0
    m, n = e
    if m >= 0
        pi = P_ss(Q)
        J = pi[m] * Q[m,n] - pi[n] * Q[n,m]
        Jplus = J + abs(J)
    end

    IR = ir_acc(m1, m2)

    (- IR + beta*EP + Jplus, IR, EP)
end

# IR FULL
function ir_full_log(logrates, args...)
    m, beta, use_mean, e = args
end