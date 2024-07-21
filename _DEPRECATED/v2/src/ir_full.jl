## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
function _idist0(idist, set_ind)
    idist0 = idist[findall(set_ind)]
    idist0 ./= sum(idist0)
    return idist0
end

export ir_full
function ir_full(weighted_edges, input_edges, c, dc, on_ind, idist, n)
    Q = build_mat(weighted_edges, input_edges, c)
    s, t = generate(n, Q, idist)
    s_in_on, rt, onrt, offrt = get_residence_time(s, t, on_ind)

    off_ind = on_ind .⊻ true
    _idist0_on = _idist0(idist, on_ind)
    _idist0_off = _idist0(idist, off_ind)

    if s_in_on
        args_ll = (on_ind, onrt, offrt, _idist0_on)
    else
        args_ll = (off_ind, offrt, onrt, _idist0_off)
    end

    # # @show args_ll
    ll_c = ll_full(weighted_edges, input_edges, c, args_ll...)
    # ll_cdc = ll_full(weighted_edges, input_edges, c + dc, args_ll...)
    # llr = cumsum(ll_c .- ll_cdc)

    # return onrt, offrt, llr
end