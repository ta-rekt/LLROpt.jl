## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# weighted_edges [(n1, n2, w), ...]
# input_edges [(n1, n2), ...]
# c
export build_mat
function build_mat(weighted_edges, input_edges, c::T) where T<:Any
    # max node index
    N = -1
    for (n1, n2, w) in weighted_edges
        N = max(N, n1, n2)
    end
    @assert N > 0
    
    # weights matrix
    Q = zeros(Float32, N, N)
    for (n1, n2, w) in weighted_edges
        Q[n1, n2] = w
    end

    # up input edges
    for (n1, n2) in input_edges
        Q[n1, n2] = Q[n1, n2] * c
    end

    # normalize
    for ri in 1:N
        Z = sum(view(Q, ri, :))
        Q[ri, ri] = -Z
    end

    return Q
end

export init_Qs!
function init_Qs!(m::LLROptModel)
    m.Q_c = build_mat(m.weighted_edges, m.input_edges, m.c)
    m.Q_cdc = build_mat(m.weighted_edges, m.input_edges, m.c + m.dc)
    return m
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export P_ss
function P_ss(d::OneBurstDecoyModel)
    A = d.Q .* 1.0
    A[:, end] .= 1.0
    b = zeros(size(A, 1))
    b[end] = 1.0
    return clamp.(A' \ b, 1e-10, 1)
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export EP_ss
function EP_ss(d::OneBurstDecoyModel)
    nstates = size(d.Q, 1)
    pss = P_ss(d)
    EP = 0.0

    for i in 1:nstates, j in 1:nstates
        if d.Q[j, i] != 0 && pss[j] > 0
            Jij = pss[i] * d.Q[i,j] - pss[j] * d.Q[j, i]
            EP += Jij * log(pss[i] * d.Q[i,j] / (pss[j] * d.Q[j, i]))
        end
    end
    return 0.5 * EP
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export Z
function Z(d::OneBurstDecoyModel)
    pss = P_ss(d)
    Pi = d.Q .* 0.0
    for ri in 1:size(d.Q, 1)
        Pi[ri, :] .= pss
    end
    return inv(Pi - d.Q) - Pi
end 

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export pi_on
# on_ind :: vector
function pi_on(d::OneBurstDecoyModel)
    return P_ss(d)' * d.ind_set
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export variance
function variance(d::OneBurstDecoyModel)
    Z_ = Z(d)
    pss = P_ss(d)
    return 2 * (d.ind_set .* pss)' * (Z_ * d.ind_set)
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export sharpness
function sharpness(d::OneBurstDecoyModel)
    # # using FiniteDiff
    # f = _c -> pi_on(d)
    # # TODO: Check stability of this
    # g = _c -> ForwardDiff.derivative(f, _c);
    # return g(c) * c
    
    # using the three-point backward finite difference

    h = d.dc


    return d_c.cr * (pi_on(d_cdc) - pi_on(d_c))/(d_cdc.cr - d_c.cr)
end

## . - -- .- .-.- . .- .-- -.- .- .- - -.-.... - -.-
export ir_acc
function ir_acc(d_c::OneBurstDecoyModel, d_cdc::OneBurstDecoyModel)
    p_c = pi_on(d_c)
    p_cdc = pi_on(d_cdc)
    var_c = variance(d_c)
    var_cdc = variance(d_cdc)

    return (p_cdc - p_c)^2 * (var_cdc + var_c) / (4 * var_c * var_cdc)
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
function _select(M, _set, _comp)
    # TODO: check views
    _M = zeros(eltype(M), length(_set), length(_comp))
    _select!(M, _M, _set, _comp)
    _M
end

export _select!
function _select!(Msrc, Mdest, _set, _comp)
    # TODO: check views
    for (i, I) in enumerate(_set), (j, J) in enumerate(_comp)
        Mdest[i, j] = Msrc[I, J]
    end
    return Mdest
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export _ddist
function _ddist(Q)
    k = size(Q, 1)
    _DD = zeros(k, k)
    for ri in 1:k, ci in 1:k
        ri == ci && continue
        _DD[ri, ci] = - Q[ri, ci]/ Q[ri, ri]
    end
    return _DD
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export  _jump_count
function _jump_count(s, on_ind)
    _jump_count = 1
    _last_is_on = on_ind[s[1]]
    for si in s
        (on_ind[si] ⊻ _last_is_on) || continue
        _jump_count += 1
        _last_is_on = on_ind[si]
    end
    _jump_count
end

