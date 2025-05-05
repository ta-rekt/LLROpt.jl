### Fisher information rate given joint ON and OFF likelihood
export integrand
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

    # if integral < 0
    #     throw("Negative integral")
    #     return nothing

    # else
    if err > 5e-2
        throw("Integral precision not met")
        return nothing

    else
        return 2 * max(integral, 0) * (d_c.cr/dc)^2 / tau
    end
end

### Fisher information rate given ON times only 
export integrandOn
function integrandOn(t_on, d_c::OneBurstDecoyModel, d_cdc::OneBurstDecoyModel)

    Pon!(t_on, d_c)
    Pon!(t_on, d_cdc)

    return d_c.Pon * (log(d_c.Pon) - log(d_cdc.Pon))
end

export FisherRateOn
function FisherRateOn(d_c::OneBurstDecoyModel, d_cdc::OneBurstDecoyModel)
    
    tauOffVal = tauOff(d_c)
    tauOnVal = tauOn(d_c)
    tau = tauOnVal + tauOffVal
    dc = d_cdc.cr - d_c.cr

    integral, err = hquadrature(t -> integrandOn(t, d_c, d_cdc), 
                0, 30/d_c.maxeigval_set; reltol=d_c.reltol, maxevals=d_c.maxevals)

    if err > 5e-2
        throw("Integral precision not met")
        return nothing

    else
        return 2 * max(integral, 0) * (d_c.cr/dc)^2 / tau
    end
end

### Fisher information rate given OFF times only 
export integrandOff
function integrandOff(t_off, d_c::OneBurstDecoyModel, d_cdc::OneBurstDecoyModel)

    Poff!(t_off, d_c)
    Poff!(t_off, d_cdc)

    return d_c.Poff * (log(d_c.Poff) - log(d_cdc.Poff))
end

export FisherRateOff
function FisherRateOff(d_c::OneBurstDecoyModel, d_cdc::OneBurstDecoyModel)
    
    tauOffVal = tauOff(d_c)
    tauOnVal = tauOn(d_c)
    tau = tauOnVal + tauOffVal
    dc = d_cdc.cr - d_c.cr

    integral, err = hquadrature(t -> integrandOff(t, d_c, d_cdc), 
                0, 30/d_c.maxeigval_comp; reltol=d_c.reltol, maxevals=d_c.maxevals)

    if err > 5e-2
        throw("Integral precision not met")
        return nothing

    else
        return 2 * max(integral, 0) * (d_c.cr/dc)^2 / tau
    end
end