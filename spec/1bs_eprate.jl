### biophysical parameters
a = 3e-3        # ums
D_tf = 7.4      # um^2/s
D_cplx = 0.1    # um^2/s
c = 5.6         # um^-3
dc = 0.1 * c
dl_tf = 4*a*D_tf
ub = 5
lb = 1e-5
max_ir_full = 0.5 * (dc/c)^2 * c*4*a*D_tf

### 1-binding site circuit parameters
edges = [(1,3), (1,2), (2,4), (2,1), (3,1), (3,4), (4,2), (4,3)]
input_edges = [(1,3), (2,4)]
on_ind = BitArray([0,1,0,1])
idist = ones(4) ./ 4
e = (1,2)                   # symmetry-breaking edge

### Bounding box on rates
llb = ones(8) .* log10(lb)
lub = ones(8) .* log10(ub)
lub[1] = log10(dl_tf)
lub[3] = log10(dl_tf)

### Optimization parameters
maxiters_acc = 30000
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
