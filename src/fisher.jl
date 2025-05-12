### Fisher information rate given joint ON and OFF likelihood
export integrand
function integrand(t, d0::OneBurstDecoyModel,
                      d1::OneBurstDecoyModel,
                      d2::OneBurstDecoyModel)

    t_off, t_on = t

    @assert d0.cr > d1.cr
    @assert d1.cr > d2.cr
    h = d0.cr - d1.cr
    @assert d2.cr ≈ d0.cr - 2*h

    Pjoint!(t_off, t_on, d0)
    Pjoint!(t_off, t_on, d1)
    Pjoint!(t_off, t_on, d2)

    dPjoint = d2.Pjoint - 4*d1.Pjoint + 3*d0.Pjoint
    dPjoint /= 2 * h

    return dPjoint^2 / d0.Pjoint
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

    integral, err = pcubature(t -> 
            integrand(t ./ [l1, l2], d0, d1, d2), 
                [eps,eps], [20, 20]; 
                reltol=d0.reltol, maxevals=d0.maxevals)

    if integral < 0
        # throw("Negative integral")
        return -1.0

    elseif err > 1e-3
        # throw("Integral precision not met")
        return -1.0

    else
        return integral*d0.cr^2 / (tau*l1*l2)
    end
end

### Fisher information rate given ON times only 
export integrandOn
function integrandOn(t_on, d0::OneBurstDecoyModel,
                           d1::OneBurstDecoyModel,
                           d2::OneBurstDecoyModel)

    @assert d0.cr > d1.cr
    @assert d1.cr > d2.cr
    h = d0.cr - d1.cr
    @assert d2.cr ≈ d0.cr - 2*h

    Pon!(t_on, d0)
    Pon!(t_on, d1)
    Pon!(t_on, d2)

    Pon = d2.Pon - 4*d1.Pon + 3*d0.Pon
    Pon /= 2 * h

    return Pon^2 / d0.Pon
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

    integral, err = pquadrature(t -> integrandOn(t / l, d0, d1, d2), 
        eps, 20; reltol=d0.reltol, maxevals=d0.maxevals)

    if integral < 0
        # throw("Negative integral")
        return -1.0

    elseif err > 1e-3
        # throw("Integral precision not met")
        return -1.0

    else
        return integral*d0.cr^2 / (tau*l)
    end
end

export integrandOff
function integrandOff(t_off, d0::OneBurstDecoyModel,
                             d1::OneBurstDecoyModel,
                             d2::OneBurstDecoyModel)
    @assert d0.cr > d1.cr
    @assert d1.cr > d2.cr
    h = d0.cr - d1.cr
    @assert d2.cr ≈ d0.cr - 2*h

    Poff!(t_off, d0)
    Poff!(t_off, d1)
    Poff!(t_off, d2)

    dPoff = d2.Poff - 4*d1.Poff + 3*d0.Poff
    dPoff /= 2 * h

    return dPoff^2 / d0.Poff
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

    integral, err = pquadrature(t -> integrandOff(t / l, d0, d1, d2),
        eps, 20;
        reltol=d0.reltol, maxevals=d0.maxevals)

    if integral < 0
        # throw("Negative integral")
        return -1.0
    elseif err > 1e-3
        # throw("Integral precision not met")
        return -1.0
    else
        return integral*d0.cr^2 / (tau*l)
    end
end