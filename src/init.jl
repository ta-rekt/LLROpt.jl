## ------
export init!
function init!(m::LLROptModel)

    init_Qs!(m)

    # indicators
    m.off_ind = m.on_ind .⊻ true
    m._idist0_on = _idist0(m.idist, m.on_ind)
    m._idist0_off = _idist0(m.idist, m.off_ind)
end

export initDecoyWeights!
function initDecoyWeights!(d::OneBurstDecoyModel)
    # transition rates
    # kboff, konb, kuon, kbon, koffu, kuoff
    kboff, konb, kuon, kbon, koffu, kuoff = 10 .^ d.logrates

    # edges = [(1, 2),    (2, 1),        (1, 6),    (6, 1),        (4, 3), (3, 4), 
    #          (4, 5),    (5, 4),        (6, 5),    (4, 1),        (2, 3), (1, 4)]
    d.weights = [kboff,     kuoff,        d.cw*kboff, d.gamma*kuoff, kbon, kuon,
                 d.cw*kbon, d.gamma*kuon, konb,       koffu,         konb]

    return d
end

export init2bsWeights!
function init2bsWeights!(d::OneBurstDecoyModel)
    # transition rates
    kboff,   kuoff,  kbon,  kuon,  koffu, kon1b, kon2b = 10 .^ d.logrates

    # edges = [(5, 1),    (1, 5),    (2, 6),    (6, 2),    (1, 2),    (1, 3), 
    #          (2, 4),    (3, 1),    (3, 4),    (4, 2),    (6, 5),    (5, 6)]
    d.weights = [2*kboff,  kuoff,     kuon,     2*kbon,    kon1b,      kboff,
                 kbon,     2*kuoff,   kon2b,    2*kuon,    koffu]

    return d
end

export initFreeWeights!
function initFreeWeights!(d::OneBurstDecoyModel)
    # transition rates
    d.weights = 10 .^ d.logrates

    return d
end

export initOneBurst!
function initOneBurst!(d::OneBurstDecoyModel)

    @assert length(d.weights) == length(d.edges)

    # building weighted adjacency matrix (rate matrix Q)
    weighted_edges = [(u,v,0.) for (u,v) in d.edges]

    for (keyb, val) in enumerate(d.weights)
        (u,v,r) = weighted_edges[keyb]
        weighted_edges[keyb] = (u,v,val)
    end

    d.Q = build_mat(weighted_edges, d.input_edges, d.cr)

    # ON and OFF partitions built from indicator arrays ind_set and ind_comp
    d.ind_comp = ones(Int, length(d.ind_set)) .- d.ind_set
    d._set = findall(isone, d.ind_set)
    d._comp = findall(isone, d.ind_comp)

    # d.Q_h_comp (fpt)
    d.Q_h_comp = d.Q .* 1.0
    for ri in d._comp
        d.Q_h_comp[ri, :] .= 0.0
    end
    UDU_comp = eigen(d.Q_h_comp)
    d.W_comp = UDU_comp.vectors
    d.W_comp_inv = inv(d.W_comp)
    d.D_comp = Diagonal(UDU_comp.values)
    d.aux_comp = zeros(size(d.D_comp)...)
    d.P_comp = zeros(size(d.Q)...)
    # exp_tD_comp = zeros(size(Q)...)

    # Q_h_set (fpt)
    d.Q_h_set = d.Q .* 1.0
    for ri in d._set
        d.Q_h_set[ri, :] .= 0.0
    end
    UDU_set = eigen(d.Q_h_set)
    d.W_set = UDU_set.vectors
    d.W_set_inv = inv(d.W_set)
    d.D_set = Diagonal(UDU_set.values)
    d.aux_set = zeros(size(d.D_comp)...)
    d.P_set = zeros(size(d.Q)...)
    d.D_aux = zeros(size(d.D_comp)...)
    # exp_tD_set = zeros(size(Q)...)

    d.F_set = zeros(size(d.Q)...)
    d.F_comp = zeros(size(d.Q)...)
    
    d.eigvals_set = UDU_set.values
    d.eigvals_comp = UDU_comp.values
    d.maxeigval_set = - maximum(x for x in d.eigvals_set if x < 0)
    d.maxeigval_comp = - maximum(x for x in d.eigvals_comp if x < 0)

    return d
end

export initDecoys!
function initDecoys!(d::OneBurstDecoyModel)

    initDecoyWeights!(d)
    initOneBurst!(d)

    return d
end

export init2bs!
function init2bs!(d::OneBurstDecoyModel)

    init2bsWeights!(d)
    initOneBurst!(d)

    return d
end

export init6bs!
function init6bs!(d::OneBurstDecoyModel)

    initFreeWeights!(d)
    initOneBurst!(d)

    return d
end

export init2bsFree!
function init2bsFree!(d::OneBurstDecoyModel)

    initFreeWeights!(d)
    initOneBurst!(d)

    return d
end