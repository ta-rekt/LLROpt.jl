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
