export Pjoint!
function Pjoint!(t_off, t_on, d::OneBurstDecoyModel)

    s1 = 1
    s3, s4, s5 = [3,4,5]

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

function integrand(t, d_c::OneBurstDecoyModel, d_cdc::OneBurstDecoyModel)

    t_off, t_on = t
    Pjoint!(t_off, t_on, d_c)
    Pjoint!(t_off, t_on, d_cdc)

    return d_c.Pjoint * (log(d_c.Pjoint) - log(d_cdc.Pjoint))
end

export FisherRate
function FisherRate(d_c::OneBurstDecoyModel, d_cdc::OneBurstDecoyModel)
    
    tauOffVal = tauOff(d_c)
    tauOnVal = tauOn(d_c)
    tau = tauOnVal + tauOffVal
    dc = d_cdc.cr - d_c.cr

    integral, err = hcubature(t -> integrand(t, d_c, d_cdc), 
                [0,0], [30/d_c.maxeigval_comp, 30/d_c.maxeigval_set]; 
                reltol=d_c.reltol, maxevals=d_c.maxevals)

    if integral < 0
        throw("Negative integral")
        return nothing

    elseif err > 1e-6
        throw("Integral precision not met")
        return nothing

    else
        return 2 * integral * (d_c.cr/dc)^2 / tau
    end

end