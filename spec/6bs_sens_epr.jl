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
lub = ones(38) .* log10(ub)
lub[1:2:24] .= log10(dl_tf)

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
betas = vcat(range(3e-4, stop=1e-5, length=15), 10 .^ range(-4.8, stop=-9, length=15))
nparams = length(betas)

# IR ACC
function ir_acc_log(logrates, args...)
    edges, iedges, c, dc, on_ind, beta, e = args
    nrates = length(edges)
    rates = 10 .^ logrates
    wedges = Vector{Tuple{Int64, Int64, Float32}}(undef, nrates)

    for i in 1:nrates
        w1 = edges[i][1]
        w2 = edges[i][2]
        w3 = rates[i]

        wedges[i] = (w1, w2, w3)
    end

    # wedges = [(u,v,10 ^ r) for ((u,v),r) in zip(edges, logrates)]

    Q = build_mat(wedges, iedges, c)
    EP = EP_ss(Q)
    
    # clockwise bias by forcing flow along edge e
    Jplus = 0
    m, n = e
    if m >= 0
        pi = P_ss(Q)
        J = pi[m] * Q[m,n] - pi[n] * Q[n,m]
        Jplus = J + abs(J)
    end

    IR = ir_acc(wedges, iedges, c, dc, on_ind)

    (- IR + beta*EP + Jplus, IR, EP)
end

# IR FULL
function ir_full_log(logrates, args...)
    m, beta, use_mean, e = args

    for (keyb, val) in enumerate(logrates)
        (u,v,r) = m.weighted_edges[keyb]
        m.weighted_edges[keyb] = (u,v,10 ^ val)
    end

    init!(m)
    EP = EP_ss(m.Q_c)
    
    # clockwise bias
    Jplus = 0
    u, v = e
    if u >= 0
        pi = P_ss(Q)
        J = pi[u] * Q[u,v] - pi[v] * Q[v,u]
        Jplus = J + abs(J)
    end

    ir_full!(m)

    # mean estimator
    if use_mean
        IR = m.llr[end] / sum(m.rt)
    
    # variance estimator
    else
        nskip = 5
        dXs = diff(m.llr[1:nskip:end-1])
        time = 2 .* cumsum(m.rt)
        sq_deltat = map(x->sqrt(x), diff(time[1:nskip:end]))
        IR = var(dXs ./ sq_deltat)
    end

    (- IR + beta*EP + Jplus, IR, EP)
end