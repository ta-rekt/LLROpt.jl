export tauOn
function tauOn(d::OneBurstDecoyModel)
    A = d.Q .* 1.0
    b = -1.0 .* ones(d.nstates,1)
    sum = 0

    # hitting probabilities
    tol = 1e-20
    d.D_aux = Diagonal(map(x -> x < -tol ? 0.0 : (abs(x) <= tol ? 1.0 : Inf), d.D_comp))
    mul!(d.aux_comp, d.W_comp, d.D_aux)
    mul!(d.P_comp, d.aux_comp, d.W_comp_inv)
    
    for k in d._set
        A[k,:] .= 0.0
        A[k,k] = 1.0
        b[k] = 0.0
    end

    mfpt_states = A \ b

    for s in d.hitting_set
        sum = sum + mfpt_states[s] * d.P_comp[d.start_state, s]
    end
    
    return sum
end


export tauOff
function tauOff(d::OneBurstDecoyModel)
    A = d.Q .* 1.0
    b = -1.0 .* ones(d.nstates, 1)
    
    for k in d._comp
        A[k,:] .= 0.0
        A[k,k] = 1.0
        b[k] = 0.0
    end
    
    mfpt_states =  A \ b

    return mfpt_states[d.start_state]
end

export Pjoint!
function Pjoint!(t_off, t_on, d::OneBurstDecoyModel)

    st = d.start_state
    hs = d.hitting_set
    sum = 0

    # fpt densities
    mul!(d.aux_comp, d.W_comp, exp(t_off * d.D_comp))
    mul!(d.P_comp, d.aux_comp, d.W_comp_inv)
    mul!(d.F_comp, d.Q_h_comp, d.P_comp)

    mul!(d.aux_set, d.W_set, exp(t_on * d.D_set))
    mul!(d.P_set, d.aux_set, d.W_set_inv)
    mul!(d.F_set, d.Q_h_set, d.P_set)

    for s in hs
        sum = sum + d.F_comp[st,s] * d.F_set[s,st] 
    end

    if sum < d.eps
        d.Pjoint = -1
    else
        d.Pjoint = sum
    end

    return d
end

export Pjoint
function Pjoint(t_off, t_on, d::OneBurstDecoyModel)

    Pjoint!(t_off, t_on, d)

    return d.Pjoint
end


export Pon!
function Pon!(t_on, d::OneBurstDecoyModel)

    # hitting probabilities
    tol = 1e-10
    d.D_aux = Diagonal(map(x -> x < -tol ? 0.0 : (abs(x) <= tol ? 1.0 : Inf), d.D_comp))
    mul!(d.aux_comp, d.W_comp, d.D_aux)
    mul!(d.P_comp, d.aux_comp, d.W_comp_inv)

    st = d.start_state
    hs = d.hitting_set
    sum = 0

    # fpt densities
    mul!(d.aux_set, d.W_set, exp(t_on * d.D_set))
    mul!(d.P_set, d.aux_set, d.W_set_inv)
    mul!(d.F_set, d.Q_h_set, d.P_set)

    for s in hs
        sum = sum + d.P_comp[st,s] * d.F_set[s,st] 
    end

    if sum < d.eps
        d.Pon = -1
    else
        d.Pon = sum
    end

    return d
end
 
export Pon
function Pon(t_on, d::OneBurstDecoyModel)

    Pon!(t_on, d)

    return d.Pon
end


export Poff!
function Poff!(t_off, d::OneBurstDecoyModel)

    st = d.start_state
    hs = d.hitting_set
    sum = 0

    # fpt densities
    mul!(d.aux_comp, d.W_comp, exp(t_off * d.D_comp))
    mul!(d.P_comp, d.aux_comp, d.W_comp_inv)
    mul!(d.F_comp, d.Q_h_comp, d.P_comp)

    for s in hs
        sum = sum + d.F_comp[st,s]
    end

    if sum < d.eps
        d.Poff = -1
    else
        d.Poff = sum
    end

    return d
end

export Poff
function Poff( t_off, d::OneBurstDecoyModel)

    Poff!(t_off, d)

    return d.Poff
end