## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export _idist0
function _idist0(idist, set_ind)
    idist0 = idist[findall(set_ind)]
    idist0 ./= sum(idist0)
    return idist0
end

export ir_full!
function ir_full!(m::LLROptModel)
    
    # s, t = generate(n, Q, idist)
    generate!(m)
    # s_in_on, rt, onrt, offrt = get_residence_time(s, t, on_ind)
    get_residence_time!(m)

    # off_ind = on_ind .⊻ true
    # _idist0_on = _idist0(idist, on_ind)
    # _idist0_off = _idist0(idist, off_ind)
    # On init!

    # TODO: add to model
    # if s_in_on
    #     args_ll = (on_ind, onrt, offrt, _idist0_on)
    # else
    #     args_ll = (off_ind, offrt, onrt, _idist0_off)
    # end

    # ll_c = ll_full(weighted_edges, input_edges, c, args_ll...)
    # ll_cdc = ll_full(weighted_edges, input_edges, c + dc, args_ll...)
    # llr = cumsum(ll_c .- ll_cdc)
    llr!(m::LLROptModel)

    # return onrt, offrt, llr
end