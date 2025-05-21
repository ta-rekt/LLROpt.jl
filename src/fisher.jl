### Fisher information rate given joint ON and OFF likelihood
export integrand
function integrand(t, d0::OneBurstDecoyModel,
                      d1::OneBurstDecoyModel,
                      d2::OneBurstDecoyModel)

    t_off, t_on = t
    u_off = t_off / (1.0 - t_off)
    u_on = t_on / (1.0 - t_on)

    @assert d0.cr > d1.cr
    @assert d1.cr > d2.cr
    h = d0.cr - d1.cr
    @assert d2.cr ≈ d0.cr - 2*h

    Pjoint!(u_off, u_on, d0)
    Pjoint!(u_off, u_on, d1)
    Pjoint!(u_off, u_on, d2)

    if d0.Pjoint < 0 || d1.Pjoint < 0 || d2.Pjoint < 0
        return 0.0
    end

    dPjoint = d2.Pjoint - 4*d1.Pjoint + 3*d0.Pjoint
    dPjoint /= 2 * h

    return dPjoint^2 / (d0.Pjoint * (1.0 - t_off)^2 * (1.0 - t_on)^2)
end

export FisherRate
function FisherRate(d0::OneBurstDecoyModel, 
                    d1::OneBurstDecoyModel,
                    d2::OneBurstDecoyModel)
    
    tauOffVal = tauOff(d0)
    tauOnVal = tauOn(d0)
    tau = tauOnVal + tauOffVal
    l1 = d0.maxeigval_comp
    l2 = d0.maxeigval_set
    eps = 1e-6

    integral, err = hcubature(t -> 
            integrand(t, d0, d1, d2), 
                [eps,eps], [1-eps, 1-eps]; 
                reltol=d0.reltol, maxevals=d0.maxevals)

    if integral < 0
        # throw("Negative integral")
        return -1.0

    elseif err > 1e-3
        # throw("Integral precision not met")
        return -1.0

    else
        return integral*d0.cr^2 / (tau)
    end
end

### Fisher information rate given ON times only 
export integrandOn
function integrandOn(t_on, d0::OneBurstDecoyModel,
                           d1::OneBurstDecoyModel,
                           d2::OneBurstDecoyModel)

    u_on = t_on / (1.0 - t_on)

    @assert d0.cr > d1.cr
    @assert d1.cr > d2.cr
    h = d0.cr - d1.cr
    @assert d2.cr ≈ d0.cr - 2*h

    Pon!(u_on, d0)
    Pon!(u_on, d1)
    Pon!(u_on, d2)

    if d0.Pon < 0 || d1.Pon < 0 || d2.Pon < 0
        return 0.0
    end

    Pon = d2.Pon - 4*d1.Pon + 3*d0.Pon
    Pon /= 2 * h

    return Pon^2 / (d0.Pon * (1.0 - t_on)^2)
end

export FisherRateOn
function FisherRateOn(d0::OneBurstDecoyModel, 
                      d1::OneBurstDecoyModel,
                      d2::OneBurstDecoyModel)
    
    tauOffVal = tauOff(d0)
    tauOnVal = tauOn(d0)
    tau = tauOnVal + tauOffVal
    l = d0.maxeigval_set
    eps = 1e-6

    integral, err = pquadrature(t -> integrandOn(t, d0, d1, d2), 
        eps, 1-eps; reltol=d0.reltol, maxevals=d0.maxevals)

    if integral < 0
        # throw("Negative integral")
        return -1.0

    elseif err > 1e-3
        # throw("Integral precision not met")
        return -1.0

    else
        return integral*d0.cr^2 / (tau)
    end
end

### Fisher information rate given OFF times only
export integrandOff
function integrandOff(t_off, d0::OneBurstDecoyModel,
                             d1::OneBurstDecoyModel,
                             d2::OneBurstDecoyModel)
    
    u_off = t_off / (1.0 - t_off)

    @assert d0.cr > d1.cr
    @assert d1.cr > d2.cr
    h = d0.cr - d1.cr
    @assert d2.cr ≈ d0.cr - 2*h

    Poff!(u_off, d0)
    Poff!(u_off, d1)
    Poff!(u_off, d2)

    if d0.Poff < 0 || d1.Poff < 0 || d2.Poff < 0
        return 0.0
    end

    dPoff = d2.Poff - 4*d1.Poff + 3*d0.Poff
    dPoff /= 2 * h

    return dPoff^2 / (d0.Poff * (1.0 - t_off)^2)
end

export FisherRateOff
function FisherRateOff(d0::OneBurstDecoyModel,
                       d1::OneBurstDecoyModel,
                       d2::OneBurstDecoyModel)
    
    tauOffVal = tauOff(d0)
    tauOnVal = tauOn(d0)
    tau = tauOnVal + tauOffVal
    l = d0.maxeigval_comp
    eps = 1e-6

    integral, err = pquadrature(t -> integrandOff(t, d0, d1, d2),
        eps, 1-eps; reltol=d0.reltol, maxevals=d0.maxevals)

    if integral < 0
        # throw("Negative integral")
        return -1.0
    elseif err > 1e-3
        # throw("Integral precision not met")
        return -1.0
    else
        return integral*d0.cr^2 / (tau)
    end
end