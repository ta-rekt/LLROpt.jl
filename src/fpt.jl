export tauOn
function tauOn(d::OneBurstDecoyModel)
    cr, cw, gamma = (d.cr, d.cw, d.gamma)
    logrates = d.logrates
    omega = cw / cr
    kboff, konb, kuon, kbon, koffu, kuoff, konu = 10 .^ logrates

    return ((konu*(konb + kuoff)*kuon*gamma*(konb + kuoff*gamma) 
            + cr^2*kboff*kbon*konb*(gamma + omega)*(konb*(1 + omega) + kuoff*(gamma + omega)) 
            + cr*(kbon*konu*(konb + kuoff)*(konb + kuoff*gamma)*(gamma + omega) 
            + kboff*konb*(konb*kuon*gamma*(1 + omega) + koffu*konb*(gamma + omega) 
            + kuoff*kuon*gamma*(gamma + omega) + koffu*kuoff*(gamma^2 + omega))))
            /(koffu*kuon*gamma*(konu*(konb + kuoff)*(konb + kuoff*gamma) 
            + cr*kboff*konb*(konb*(1 + omega) + kuoff*(gamma + omega)))))
end


export tauOff
function tauOff(d::OneBurstDecoyModel)
    cr, cw, gamma = (d.cr, d.cw, d.gamma)
    logrates = d.logrates
    omega = cw / cr
    kboff, konb, kuon, kbon, koffu, kuoff, konu = 10 .^ logrates

    numerator = (konb + kuoff)*(konb + kuoff*gamma) + cr*kboff*(konb*(1 + omega) + kuoff*(gamma + omega))
    denominator = (konu*(konb + kuoff)*(konb + kuoff*gamma) 
                   + cr*kboff*konb*(konb*(1 + omega) + kuoff*(gamma + omega)))
     
    return numerator / denominator

end

export Pjoint!
function Pjoint!(t_off, t_on, d::OneBurstDecoyModel)

    s1 = 1
    s3, s4, s5 = [3,4,5]

    # fpt densities
    mul!(d.aux_comp, d.W_comp, exp(t_off * d.D_comp))
    mul!(d.P_comp, d.aux_comp, d.W_comp_inv)
    mul!(d.F_comp, d.Q_h_comp, d.P_comp)

    mul!(d.aux_set, d.W_set, exp(t_on * d.D_set))
    mul!(d.P_set, d.aux_set, d.W_set_inv)
    mul!(d.F_set, d.Q_h_set, d.P_set)

    d.Pjoint = max((d.F_comp[s1,s3] * d.F_set[s3,s1] 
                  + d.F_comp[s1,s4] * d.F_set[s4,s1] 
                  + d.F_comp[s1,s5] * d.F_set[s5,s1]), d.eps)

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

    s1 = 1
    s3, s4, s5 = [3,4,5]

    # fpt densities
    mul!(d.aux_set, d.W_set, exp(t_on * d.D_set))
    mul!(d.P_set, d.aux_set, d.W_set_inv)
    mul!(d.F_set, d.Q_h_set, d.P_set)

    d.Pon = max((d.P_comp[s1,s3] * d.F_set[s3,s1] 
               + d.P_comp[s1,s4] * d.F_set[s4,s1] 
               + d.P_comp[s1,s5] * d.F_set[s5,s1]), d.eps)

    return d
end

export Pon
function Pon(t_on, d::OneBurstDecoyModel)

    Pon!(t_on, d)

    return d.Pon
end


export Poff!
function Poff!(t_off, d::OneBurstDecoyModel)

    s1 = 1
    s3, s4, s5 = [3,4,5]

    # fpt densities
    mul!(d.aux_comp, d.W_comp, exp(t_off * d.D_comp))
    mul!(d.P_comp, d.aux_comp, d.W_comp_inv)
    mul!(d.F_comp, d.Q_h_comp, d.P_comp)

    d.Poff = max((d.F_comp[s1,s3] + d.F_comp[s1,s4] + d.F_comp[s1,s5]), d.eps)

    return d
end

export Poff
function Poff( t_off, d::OneBurstDecoyModel)

    Poff!(t_off, d)

    return d.Poff
end