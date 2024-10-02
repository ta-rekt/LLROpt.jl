a = 3e-3        # um
D_tf = 7.4      # um^2/s
D_cplx = 0.1    # um^2/s
cr = 5.6        # um^-3
dc = 0.1 * cr
dl_tf = 4*a*D_tf
ub = 5
lb = 1e-5
max_ir_full = 0.5 * (dc/cr)^2 * (cr*dl_tf*ub)/(cr*dl_tf + ub)

# upper bounds for pure ligand circuit
llb1 = ones(8) .* log10(lb)
lub1 = ones(8) .* log10(ub)
lub1[1] = log10(dl_tf)
lub1[3] = log10(dl_tf)

# upper bounds for mixed ligands circuit
llb2 = ones(14) .* log10(lb)
lub2 = ones(14) .* log10(ub)
lub2[2] = log10(dl_tf)
lub2[3] = log10(dl_tf)
lub2[6] = log10(dl_tf)
lub2[7] = log10(dl_tf)

# circuit descriptions
edges1 = [(1,3), (1,2), (2,4), (2,1), (3,1), (3,4), (4,2), (4,3)]
edges2 = [(5,1), (1,5), (2,6), (6,2), (1,2), (1,3), (2,4),
          (2,1), (3,1), (3,4), (4,2), (4,3), (5,6), (6,5)]
input_edges = [(1,3), (2,4)]
on_ind1 = BitArray([0,1,0,1])
on_ind2 = BitArray([0,1,0,1,0,1])

# GCMAES params
maxiters_acc = 10000
maxiters_full = 5000
xinit1 = (lub1 .+ llb1) ./ 2
xinit2 = (lub2 .+ llb2) ./ 2
we1 = [(u,v,r) for ((u,v),r) in zip(edges1, xinit1)]
we2 = [(u,v,r) for ((u,v),r) in zip(edges2, xinit2)]
n = 500000
N = 100

whichf = "acc"