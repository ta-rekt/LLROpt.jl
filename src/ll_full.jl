# . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export ll_full
function ll_full(Q, ind_set, t_set, t_comp, idist_set)
    # @assert length(t_set) == length(t_comp)
    ind_comp = ones(Int, length(ind_set)) .- ind_set
    _set = findall(isone, ind_set)
    _comp = findall(isone, ind_comp)

    w_set = zeros(1, length(_set))
    w_comp = zeros(1, length(_comp))
    w_set .= idist_set' # first init
    wv = zeros(length(t_set) + length(t_comp) + 1)
    wv[1] = log(sum(idist_set))
    i = 2

    # Q_h_comp (fpt)
    Q_h_comp = Q .* 1.0
    for ri in _comp
        Q_h_comp[ri, :] .= 0.0
    end
    UDU_comp = eigen(Q_h_comp)
    W_comp = UDU_comp.vectors
    W_comp_inv = inv(W_comp)
    D_comp = Diagonal(UDU_comp.values)
    aux_comp = zeros(size(D_comp)...)
    P_comp = zeros(size(Q)...)
    exp_tD_comp = zeros(size(Q)...)

    # Q_h_set (fpt)
    Q_h_set = Q .* 1.0
    for ri in _set
        Q_h_set[ri, :] .= 0.0
    end
    UDU_set = eigen(Q_h_set)
    W_set = UDU_set.vectors
    W_set_inv = inv(W_set)
    D_set = Diagonal(UDU_set.values)
    aux_set = zeros(size(D_comp)...)
    P_set = zeros(size(Q)...)
    exp_tD_set = zeros(size(Q)...)

    R = zeros(size(Q)...)
    S = zeros(size(Q)...)
    _R = zeros(eltype(Q), length(_set), length(_comp))
    _S = zeros(eltype(Q), length(_comp), length(_set))

    # for (ts, tc) in zip(t_set, t_comp)
    for (ts, tc) in zip(t_set, t_comp)
        # P = W * exp(t * D) * inv(W)
        # R = Q_h_comp * P(UDU_comp, ts)
        mul!(aux_comp, W_comp, exp(ts * D_comp))
        mul!(P_comp, aux_comp, W_comp_inv)
        mul!(R, Q_h_comp, P_comp)

        # P = W * exp(t * D) * inv(W)
        # S = Q_h_set * P(UDU_set, tc)
        
        # mul!(exp_tD_set, tc, D_set)
        # exp!(exp_tD_set)
        # mul!(aux_set, W_set, exp_tD_set)
        # mul!(aux_set, W_set, exp(exp_tD_set))

        mul!(aux_set, W_set, exp(tc * D_set))
        mul!(P_set, aux_set, W_set_inv)
        mul!(S, Q_h_set, P_set)
        
        _select!(R, _R, _set, _comp)
        _select!(S, _S, _comp, _set)

        # TODO: do it inline
        w_set ./= sum(w_set)
        # w_comp = (w_set' * _R)'
        mul!(w_comp, w_set, _R)
        wv[i] = log(sum(w_comp)); i+=1

        w_comp ./= sum(w_comp)
        mul!(w_set, w_comp, _S)
        # w_set = ((w_comp ./ sum(w_comp))' * _S)'
        wv[i] = log(sum(w_set)); i+=1
    end
    wv
    
end

export llr!
function llr!(m::LLROptModel)
    # select based on m.s_in_on
    m.ind_set = m.s_in_on ? m.on_ind : m.off_ind
    m.t_set = m.s_in_on ? m.onrt : m.offrt
    m.t_comp = m.s_in_on ? m.offrt : m.onrt
    m.idist_set = m.s_in_on ? m._idist0_on : m._idist0_off
    
    # TODO: inline
    m.ll_c = ll_full(m.Q_c, m.ind_set, m.t_set, m.t_comp, m.idist_set)
    m.ll_cdc = ll_full(m.Q_cdc, m.ind_set, m.t_set, m.t_comp, m.idist_set)
    m.llr = cumsum(m.ll_c .- m.ll_cdc)
    return m
end
