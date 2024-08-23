import Pkg
Pkg.add("JSON")
using JSON
using LLROpt

@time let 
    a = 3e-3        # um
    D_tf = 7.4      # um^2/s
    D_cplx = 0.1    # um^2/s
    w = 0.2
    cr = 5.6         # um^-3
    dc = 0.1 * cr
    cw = w * cr
    dl_tf = 4*a*D_tf
    ub = 5
    lb = 1e-5
    max_ir_full = 0.5 * (dc/cr)^2 * (cr*dl_tf*ub)/(cr*dl_tf + ub)

    kboff = 0.0888
    konu = 1e-5
    kbon = 1e-5
    koffu = 5
    kuoff = 1e-5
    konb = 5
    kuon = 5
    koffb = 1e-5

    n = 10000000
    nvals = 20
    ir_full_pure = Vector{Float32}(undef, nvals)
    ir_acc_pure = Vector{Float32}(undef, nvals)
    ep_pure = Vector{Float32}(undef, nvals)
    ir_full_mix = Vector{Float32}(undef, nvals)
    ir_acc_mix = Vector{Float32}(undef, nvals)
    ep_mix = Vector{Float32}(undef, nvals)

    # wvals = range(0.0001, stop=1, length=nvals)
    wvals = [1]
    gammas = 10 .^ range(0, 4, nvals)

    wedges2 = [(1,3,kboff), 
                (1,2,konu), 
                (2,4,kbon), 
                (2,1,koffu), 
                (3,1,kuoff), 
                (3,4,konb), 
                (4,2,kuon), 
                (4,3,koffb)]

    iedges = [(1,3), (2,4)]
    on_ind1 = BitArray([0,1,0,1,0,1])
    on_ind2 = BitArray([0,1,0,1])

    m2 = LLROptModel(;
        weighted_edges = wedges2,
        input_edges = iedges,
        on_ind = on_ind2,
        c = cr,
        dc = dc,
        n = n,
        idist = ones(4) ./ 4,
    )
    init!(m2)

    for (keyw, w) in enumerate(wvals)
        for (keyg, gamma) in enumerate(gammas)

            cw = w*cr
            wedges1 = [(5,1,gamma*kuoff),
                        (1,5,cw*kboff),
                        (2,6,cw*kbon),
                        (6,2,gamma*kuon),
                        (1,2,konu), 
                        (1,3,kboff), 
                        (2,4,kbon), 
                        (2,1,koffu), 
                        (3,1,kuoff), 
                        (3,4,konb), 
                        (4,2,kuon), 
                        (4,3,koffb),
                        (5,6,konb),
                        (6,5,koffb)]

            m1 = LLROptModel(;
                weighted_edges = wedges1,
                input_edges = iedges,
                on_ind = on_ind1,
                c = cr,
                dc = dc,
                n = n,
                idist = ones(6) ./ 6,
            )

            init!(m1)
            ir_full!(m1)
            ir_full!(m2)
            ir_full_mix[keyg] = m1.llr[end] / sum(m1.rt)
            ir_full_pure[keyg] = m2.llr[end] / sum(m2.rt)

            Q1 = build_mat(wedges1, iedges, cr)
            Q2 = build_mat(wedges2, iedges, cr)
            ep_mix[keyg] = EP_ss(Q1)
            ep_pure[keyg] = EP_ss(Q2)

            ir_acc_mix[keyg] = ir_acc(wedges1, iedges, cr, dc, on_ind1)
            ir_acc_pure[keyg] = ir_acc(wedges2, iedges, cr, dc, on_ind2)
        end
    end

    output = Dict([("ir_full_mix", ir_full_mix),
                    ("ir_full_pure", ir_full_pure),
                    ("ep_mix", ep_mix),
                    ("ep_pure", ep_pure),
                    ("ir_acc_mix", ir_acc_mix),
                    ("ir_acc_pure", ir_acc_pure),
                    ("wvals", wvals)])

    file1 = "data/specificity_output_20240730.json"

    open(file1, "w") do f
        JSON.print(f, output, 4)
    end

end