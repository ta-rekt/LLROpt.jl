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

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export P
function P(Q, t)
    UDU = eigen(Q)
    W = UDU.vectors
    v = UDU.values
    D = Diagonal(v)
    P = W * exp(t * D) * inv(W)
    return P
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# hset: vector
export fpt
function fpt(Q, hset, t)
    Q_h = Q .* 1.0
    for ri in hset
        Q_h[ri, :] .= 0.0
    end
    return Q_h * P(Q_h, t)
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
    for (i, I) in enumerate(_set), (j, J) in enumerate(_comp)
        _M[i, j] = M[I, J]
    end
    _M
end

# . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export ll_full
function ll_full(weighted_edges, input_edges, c, ind_set, t_set, t_comp, idist_set)
    # @assert length(t_set) == length(t_comp)
    Q = build_mat(weighted_edges, input_edges, c)
    ind_comp = ones(Int, length(ind_set)) .- ind_set
    _set = findall(isone, ind_set)
    _comp = findall(isone, ind_comp)

    w = copy(idist_set)
    wv = zeros(length(t_set) + length(t_comp) + 1)
    wv[1] = log(sum(idist_set))
    i = 2
    for (ts, tc) in zip(t_set, t_comp)
        R = fpt(Q, _comp, ts)
        S = fpt(Q, _set, tc)
        # @show Q
        # @show _set
        # @show S
        _R = _select(R, _set, _comp)
        # @show _R
        _S = _select(S, _comp, _set)
        # @show _S
        # TODO: do it inline
        w = ((w ./ sum(w))' * _R)'
        wv[i] = log(sum(w)); i+=1
        w = ((w ./ sum(w))' * _S)'
        wv[i] = log(sum(w)); i+=1
    end
    return wv
    
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

# . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# TODO: Finish this shit
export generate
function generate(n, Q, idist)
    s, t = zeros(Int, n), zeros(n)
    
    s[1] = sample(eachindex(idist), Weights(idist))
    q = -Q[s[1], s[1]]

    _DD = _ddist(Q)

    for i in 1:n-1
        si = s[i]
        q = -Q[si, si]

        # t
        e = Exponential(1/q)
        t[i] = rand(e)

        # s        
        _dd = _DD[si, :]
        s[i+1] = sample(eachindex(_dd), Weights(_dd))
    end

    # t
    si = s[end]
    q = -Q[si, si]
    e = Exponential(1/q)
    t[end] = rand(e)

    return s, t
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

# . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
export get_residence_time
function get_residence_time(s, t, on_ind)
    
    jc = _jump_count(s, on_ind)
    rt = zeros(Float64, jc)

    _last_is_on = on_ind[s[1]]

    ji = 1 # jump index
    for (si, ti) in zip(s, t)
        if (on_ind[si] ⊻ _last_is_on)
            _last_is_on = on_ind[si]
            ji += 1
        end
        rt[ji] += ti
    end
    # postprocess
    s_in_on = on_ind[s[1]]
    onrt = s_in_on ? rt[1:2:end] : rt[2:2:end]
    offrt = s_in_on ? rt[2:2:end] : rt[1:2:end]
    return s_in_on, rt, onrt, offrt
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
function _idist0(idist, set_ind)
    idist0 = idist[findall(set_ind)]
    idist0 ./= sum(idist0)
    return idist0
end

function ir_full(weighted_edges, input_edges, c, dc, on_ind, idist, n)
    Q = build_mat(weighted_edges, input_edges, c)
    s, t = generate(n, Q, idist)
    s_in_on, rt, onrt, offrt = get_residence_time(s, t, on_ind)

    off_ind = on_ind .⊻ true
    _idist0_on = _idist0(idist, on_ind)
    _idist0_off = _idist0(idist, off_ind)

    if s_in_on
        args_ll = (on_ind, onrt, offrt, _idist0_on)
    else
        args_ll = (off_ind, offrt, onrt, _idist0_off)
    end

    # @show args_ll
    ll_c = ll_full(weighted_edges, input_edges, c, args_ll...)
    ll_cdc = ll_full(weighted_edges, input_edges, c + dc, args_ll...)
    llr = cumsum(ll_c .- ll_cdc)

    return onrt, offrt, llr
end