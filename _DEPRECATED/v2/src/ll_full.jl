# . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export ll_full
function ll_full(weighted_edges, input_edges, c, ind_set, t_set, t_comp, idist_set)
    # @assert length(t_set) == length(t_comp)
    Q = build_mat(weighted_edges, input_edges, c)
    ind_comp = ones(Int, length(ind_set)) .- ind_set
    _set = findall(isone, ind_set)
    _comp = findall(isone, ind_comp)

    w_set = zeros(1, length(_set))
    w_comp = zeros(1, length(_comp))
    w_set .= idist_set # first init
    wv = zeros(length(t_set) + length(t_comp) + 1)
    wv[1] = log(sum(idist_set))
    i = 2

    # Q_h_comp (fpt)
    Q_h_comp = Q .* 1.0
    for ri in _comp
        Q_h_comp[ri, :] .= 0.0
    end
    UDU_comp = eigen(Q_h_comp)

    # Q_h_set (fpt)
    Q_h_set = Q .* 1.0
    for ri in _set
        Q_h_set[ri, :] .= 0.0
    end
    UDU_set = eigen(Q_h_set)

    _R = zeros(eltype(Q), length(_set), length(_comp))
    _S = zeros(eltype(Q), length(_comp), length(_set))

    for (ts, tc) in zip(t_set, t_comp)

        R = Q_h_comp * P(UDU_comp, ts)
        S = Q_h_set * P(UDU_set, tc)
        
        _select!(R, _R, _set, _comp)
        _select!(S, _S, _comp, _set)

        # TODO: do it inline
        # w = ((w ./ sum(w))' * _R)'
        # wv[i] = log(sum(w)); i+=1
        # w = ((w ./ sum(w))' * _S)'
        # wv[i] = log(sum(w)); i+=1

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