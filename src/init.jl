## ------
export init!
function init!(m::LLROptModel)

    init_Qs!(m)

    # indicators
    m.off_ind = m.on_ind .⊻ true
    m._idist0_on = _idist0(m.idist, m.on_ind)
    m._idist0_off = _idist0(m.idist, m.off_ind)

end

export initOneBurst!
function initOneBurst!(d::OneBurstDecoyModel)

    # transition rates
    kboff, konb, kuon, kbon, koffu, kuoff, konu = 10 .^ d.logrates

    ratesFull = [kboff,     kuoff,        d.cw*kboff, d.gamma*kuoff, kbon, kuon,
                 d.cw*kbon, d.gamma*kuon, konb,       koffu,         konb, konu]

    # building weighted adjacency matrix (rate matrix Q)
    weighted_edges = [(u,v,0.) for (u,v) in d.edges]

    for (keyb, val) in enumerate(ratesFull)
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
    # exp_tD_set = zeros(size(Q)...)

    d.F_set = zeros(size(d.Q)...)
    d.F_comp = zeros(size(d.Q)...)
    
    d.eigvals_set = UDU_set.values
    d.eigvals_comp = UDU_comp.values
    d.maxeigval_set = - maximum(x for x in d.eigvals_set if x < 0)
    d.maxeigval_comp = - maximum(x for x in d.eigvals_comp if x < 0)

    return d
end