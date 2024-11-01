@time begin
    using LLROpt
    using LinearAlgebra
    using ForwardDiff
    using BenchmarkTools
    using Distributions
    using StatsBase
    using Random
    using Base.Threads
    using Statistics
    using JSON
end

########## generate

function generate(n, Q, idist)

    s, t = zeros(Int, n), zeros(n)
    
    s[1] = sample(eachindex(idist), Weights(idist))
    q = -Q[s[1], s[1]]

    _DD = _ddist(Q)

    for i in 1:n-1
        si = s[i]
        q = -Q[si, si]

        # t
        e = Exponential(1/q)
        t[i] = rand(e)

        # s        
        _dd = _DD[si, :]
        s[i+1] = sample(eachindex(_dd), Weights(_dd))
    end

    # t
    si = s[end]
    q = -Q[si, si]
    e = Exponential(1/q)
    t[end] = rand(e)

    return s, t
end


########### get residence time

function get_residence_time(s, t, on_ind)
    
    jc = _jump_count(s, on_ind)
    rt = zeros(Float64, jc)

    _last_is_on = on_ind[s[1]]

    ji = 1 # jump index
    for (si, ti) in zip(s, t)
        if (on_ind[si] ⊻ _last_is_on)
            _last_is_on = on_ind[si]
            ji += 1
        end
        rt[ji] += ti
    end
    # postprocess
    s_in_on = on_ind[s[1]]
    onrt = s_in_on ? rt[1:2:end] : rt[2:2:end]
    offrt = s_in_on ? rt[2:2:end] : rt[1:2:end]
    return s_in_on, rt, onrt, offrt
end


############# LLR

function ll_full(Q, ind_set, t_set, t_comp, idist_set)
    ind_comp = ones(Int, length(ind_set)) .- ind_set
    _set = findall(isone, ind_set)
    _comp = findall(isone, ind_comp)

    w_set = zeros(1, length(_set))
    w_comp = zeros(1, length(_comp))
    w_set .= idist_set' # first init
    wv = zeros(length(t_set) + length(t_comp) + 1)
    wv[1] = log(sum(idist_set))
    i = 2

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

    for (ts, tc) in zip(t_set, t_comp)
        mul!(aux_comp, W_comp, exp(ts * D_comp))
        mul!(P_comp, aux_comp, W_comp_inv)
        mul!(R, Q_h_comp, P_comp)

        mul!(aux_set, W_set, exp(tc * D_set))
        mul!(P_set, aux_set, W_set_inv)
        mul!(S, Q_h_set, P_set)
        
        _select!(R, _R, _set, _comp)
        _select!(S, _S, _comp, _set)

        w_set ./= sum(w_set)
        mul!(w_comp, w_set, _R)
        wv[i] = log(sum(w_comp)); i+=1

        w_comp ./= sum(w_comp)
        mul!(w_set, w_comp, _S)
        wv[i] = log(sum(w_set)); i+=1
    end
    wv
    
end

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

@time let

    #### merge the above, add stopping mechanism, keep it fast


end