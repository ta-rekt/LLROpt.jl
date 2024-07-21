## ------
export init!
function init!(m::LLROptModel)

    init_Qs!(m)

    # indicators
    m.off_ind = m.on_ind .⊻ true
    m._idist0_on = _idist0(m.idist, m.on_ind)
    m._idist0_off = _idist0(m.idist, m.off_ind)

end
