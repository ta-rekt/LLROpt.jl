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
    Q = zeros(T, N, N)
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
function P_ss(Q)
    A = Q .* 1.0
    A[:, end] .= 1.0
    b = zeros(size(A, 1))
    b[end] = 1.0
    return A' \ b
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export EP_ss
function EP_ss(Q)
    nstates = size(Q, 1)
    pi = P_ss(Q)
    EP = 0.0

    for i in 1:nstates, j in 1:nstates
        if Q[j, i] != 0 && pi[j] > 0
            Jij = pi[i] * Q[i,j] - pi[j] * Q[j, i]
            EP += Jij * log(pi[i] * Q[i,j] / (pi[j] * Q[j, i]))
        end
    end
    return 0.5 * EP
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export Z
function Z(Q)
    pi = P_ss(Q)
    Pi = Q .* 0.0
    for ri in 1:size(Q, 1)
        Pi[ri, :] .= pi
    end
    return inv(Pi - Q) - Pi
end 

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export pi_on
# on_ind :: vector
function pi_on(weighted_edges, input_edges, c, on_ind)
    Q = build_mat(weighted_edges, input_edges, c)
    return P_ss(Q)' * on_ind
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export variance
function variance(weighted_edges, input_edges, c, on_ind)
    Q = build_mat(weighted_edges, input_edges, c)
    Z_ = Z(Q)
    pi = P_ss(Q)
    return 2 * (on_ind .* pi)' * (Z_ * on_ind)
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export sharpness
function sharpness(weighted_edges, input_edges, c, on_ind)
    # using FiniteDiff
    f = _c -> pi_on(weighted_edges, input_edges, _c, on_ind)
    # TODO: Check stability of this
    g = _c -> ForwardDiff.derivative(f, _c);
    return g(c) * c
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export ir_acc
function ir_acc(weighted_edges, input_edges, c, dc, on_ind)
    p_c = pi_on(weighted_edges, input_edges, c, on_ind)
    p_cdc = pi_on(weighted_edges, input_edges, c + dc, on_ind)
    var_c = variance(weighted_edges, input_edges, c, on_ind)
    var_cdc = variance(weighted_edges, input_edges, c + dc, on_ind)

    return (p_cdc - p_c)^2 * (var_cdc + var_c) / (4 * var_c * var_cdc)
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
function _select(M, _set, _comp)
    # TODO: check views
    _M = zeros(eltype(M), length(_set), length(_comp))
    _select!(M, _M, _set, _comp)
    _M
end

function _select!(Msrc, Mdest, _set, _comp)
    # TODO: check views
    for (i, I) in enumerate(_set), (j, J) in enumerate(_comp)
        Mdest[i, j] = Msrc[I, J]
    end
    return Mdest
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
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

