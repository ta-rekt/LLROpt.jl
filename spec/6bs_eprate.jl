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

input_edges = [(i,i+1) for i in range(1,13) if i != 7]
on_ind = vcat(zeros(Bool, 7), ones(Bool, 7))
on_ind = BitArray(on_ind)
idist = ones(14) ./ 14

### Bounding box on rates
ub = 5
lb = 1e-3
llb = ones(38) .* log10(lb)
lub = ones(38) .* log10(ub)
lub[1:2:24] .= log10(dl_tf)

### Optimization parameters
maxiters_acc = 20000
maxiters_full = 5000
sigma0 = log10(ub) / 4
x0 = (lub .+ llb) ./ 2
we = [(u,v,r) for ((u,v),r) in zip(edges, x0)]
n = 1000000
k = 1
N = 2000
whichf = "acc"

### temperature parameter
betas = vcat(range(3e-4, stop=1e-5, length=20), 10 .^ range(-4.8, stop=-9, length=20))
nparams = length(betas)