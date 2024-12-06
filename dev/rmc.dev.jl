## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
@time begin
    using LLROpt
    using LinearAlgebra
    using ForwardDiff
    using BenchmarkTools
    using Distributions
    using StatsBase
    using Random
    using Base.Threads
    using JSON
    using Statistics
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: implementing the RMC algorithm from J. Vanneste PRE 81, 036701 (2010)

function generate_rmc(n, k, Q, idist)

    s, t = zeros(Int, n, k), zeros(Float32, n, k)
    
    for j in 1:k
        s[1,j] = sample(eachindex(idist), Weights(idist))
        q = -Q[s[1,j], s[1,j]]

        _DD = _ddist(Q)

        for i in 1:n-1
            si = s[i,j]
            q = -Q[si, si]

            # t
            e = Exponential(1/q)
            t[i,j] = rand(e)

            # s        
            _dd = _DD[si, :]
            s[i+1,j] = sample(eachindex(_dd), Weights(_dd))
        end

        # t
        si = s[end,j]
        q = -Q[si, si]
        e = Exponential(1/q)
        t[end,j] = rand(e)
    end

    return s, t
end

function generate_rmc!(m::LLROptModel)
    # TODO: inline
    s, t = generate_rmc(m.n, m.k, m.Q_c, m.idist)
    
    m.s = s
    m.t = t
    return m
end


function get_residence_time_rmc(s, t, on_ind)
    
    n, k = size(s)
    jc = _jump_count(s[1:end, 1], on_ind)
    s_in_on = zeros(Bool, k)
    bc = zeros(Int, k)
    rt = zeros(Float32, jc*2, k)
    onrt = zeros(Float32, jc*2, k)
    offrt = zeros(Float32, jc*2, k)

    for j in 1:k
        _last_is_on = on_ind[s[1,j]]
        s_in_on[j] = _last_is_on   # jth trajectory started in on

        ji = 1 # jump index (counts ON->OFF and OFF->ON transition)
        bi = 0 # burst index (counts ON->ON or OFF->OFF cycles)

        for i in 1:n
            if (on_ind[s[i,j]] ⊻ _last_is_on)   # partition has changed
                _last_is_on = on_ind[s[i,j]]

                if _last_is_on
                    offrt[bi+1,j] = rt[ji, j]
                else
                    onrt[bi+1,j] = rt[ji, j]
                end

                if (_last_is_on === s_in_on[j]) && ji > 1  # a cycle is completed
                    bi += 1
                end

                ji += 1
            end
            rt[ji, j] += t[i,j]
        end

        bc[j] = bi
    end

    return s_in_on, rt, onrt, offrt, bc
end

function get_residence_time_rmc!(m::LLROptModel)
    s_in_on, rt, onrt, offrt, bc = get_residence_time_rmc(m.s, m.t, m.on_ind)
    m.s_in_on = s_in_on
    m.rt = rt
    m.bc = bc
    m.onrt = onrt
    m.offrt = offrt
    return m
end


# assumes all trajectories start in the ON partition
function ll_full_rmc(Q, on_ind, idist, bc, t_set, t_comp)

    # @assert length(t_set) == length(t_comp)

    #### array initialization ####

    off_ind = on_ind .⊻ true
    _on = findall(isone, on_ind)
    _off = findall(isone, off_ind)
    _idist_on = _idist0(idist, on_ind)

    # Q_h_off (fpt)
    Q_h_off = Q .* 1.0
    for ri in _off
        Q_h_off[ri, :] .= 0.0
    end
    UDU_off = eigen(Q_h_off)
    W_off = UDU_off.vectors
    W_off_inv = inv(W_off)
    D_off = Diagonal(UDU_off.values)
    aux_off = zeros(size(D_off)...)
    P_off = zeros(size(Q)...)

    # Q_h_on (fpt)
    Q_h_on = Q .* 1.0
    for ri in _on
        Q_h_on[ri, :] .= 0.0
    end
    UDU_on = eigen(Q_h_on)
    W_on = UDU_on.vectors
    W_on_inv = inv(W_on)
    D_on = Diagonal(UDU_on.values)
    aux_on = zeros(size(D_off)...)
    P_on = zeros(size(Q)...)

    R = zeros(size(Q)...)
    S = zeros(size(Q)...)
    _R = zeros(eltype(Q), length(_on), length(_off))
    _S = zeros(eltype(Q), length(_off), length(_on))

    n = minimum(bc)      # smallest number of on-off cycles
    k = length(bc)   # number of trajectories

    w_on = zeros(1, length(_on))
    w_off = zeros(1, length(_off))
    w_on .= _idist_on'   # first init
    wv = zeros(Float64, k, 2*n+1)    # log vector norms
    vec = zeros(Float64, k, length(w_on))
    vec_sampled = zeros(Float64, k, length(w_on))
    alpha = zeros(Float64, k)
    logbeta = zeros(Float64, n)
    jsampl = zeros(Int32, k)
    t_on, t_off = 0.0, 0.0

    #### trajectory-dependent operations ####

    for j in 1:k
        wv[j,1] = sum(w_on)
        vec[j,:] = w_on
        vec_sampled[j,:] = w_on
    end

    i = 1
    jmp = 2

    while i <= n
        beta = 0
        jsampl = 0

        for j in 1:k
            t_on, t_off = t_set[i,j], t_comp[i,j]

            mul!(aux_off, W_off, exp(t_on * D_off))
            mul!(P_off, aux_off, W_off_inv)
            mul!(R, Q_h_off, P_off)

            mul!(aux_on, W_on, exp(t_off * D_on))
            mul!(P_on, aux_on, W_on_inv)
            mul!(S, Q_h_on, P_on)
            
            _select!(R, _R, _on, _off)
            _select!(S, _S, _off, _on)

            w_on = vec_sampled[j,:]'
            alpha[j] = sum(w_on)
            w_on ./= sum(w_on)
            mul!(w_off, w_on, _R)
            wv[j,jmp] = log(sum(w_off));

            w_off ./= sum(w_off)
            mul!(w_on, w_off, _S)
            wv[j,jmp+1] = log(sum(w_on));
            vec[j,:] = w_on'

            beta += alpha[j]
        end

        alpha ./= beta
        logbeta[i] = log(beta)
        jsampl = sample(eachindex(alpha), Weights(alpha), k)

        for j in 1:k
            vec_sampled[j,:] = vec[jsampl[j],:]
        end

        i += 1
        jmp += 2
    end

    # return log sum of betas - logK
    sum(logbeta)/n - log(k)
end

function llr_rmc!(m::LLROptModel)
    # TODO: inline
    m.ll_c = ll_full_rmc(m.Q_c, m.on_ind, m.idist, m.bc, m.onrt, m.offrt)
    m.ll_cdc = ll_full_rmc(m.Q_cdc, m.on_ind, m.idist, m.bc, m.onrt, m.offrt)
    m.llr = m.ll_c .- m.ll_cdc
    return m
end

function ir_full_rmc!(m::LLROptModel)
    generate_rmc!(m)
    get_residence_time_rmc!(m)
    llr_rmc!(m::LLROptModel)
end

@time let 
    date = 20241104
    c = 5.6
    kb = 0.0888
    kon = 5
    ku = 5
    koff = 5
    n = 1000
    k = 1000
    dt = (1/(c*kb) + 1/kon + 1/ku + 1/koff)

    global m = LLROptModel(;
        weighted_edges = [(1,3,kb), 
                          (1,2,1e-5), 
                          (2,4,1e-5), 
                          (2,1,koff), 
                          (3,1,1e-5), 
                          (3,4,kon), 
                          (4,2,ku), 
                          (4,3,1e-5)],
        input_edges = [(1,3), (2,4)],
        on_ind = BitArray([0,1,0,1]),
        c = c,
        dc = c / 10,
        n = n,
        k = k,
        idist = [0, 0.5, 0, 0.5]
    )
    init!(m)

    #### RMC
    generate_rmc!(m)
    get_residence_time_rmc!(m)
    ll_c = ll_full_rmc(m.Q_c, m.on_ind, m.idist, m.bc, m.onrt, m.offrt)
    ll_cdc = ll_full_rmc(m.Q_cdc, m.on_ind, m.idist, m.bc, m.onrt, m.offrt)
    (ll_c - ll_cdc)/dt
end

@time let 
    date = 20241104
    c = 5.6
    kb = 0.0888
    kon = 5
    ku = 5
    koff = 5
    n = 1000000
    k = 1
    dt = (1/(c*kb) + 1/kon + 1/ku + 1/koff)

    global m = LLROptModel(;
        weighted_edges = [(1,3,kb), 
                          (1,2,1e-5), 
                          (2,4,1e-5), 
                          (2,1,koff), 
                          (3,1,1e-5), 
                          (3,4,kon), 
                          (4,2,ku), 
                          (4,3,1e-5)],
        input_edges = [(1,3), (2,4)],
        on_ind = BitArray([0,1,0,1]),
        c = c,
        dc = c / 10,
        n = n,
        k = k,
        idist = [0, 0.5, 0, 0.5]
    )
    init!(m)

    #### Brute-force
    ir_full!(m)
    m.llr[end] / sum(m.rt)
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing the Jensen gap

function l_full(Q, ind_set, t_set, t_comp, idist_set)
    ind_comp = ones(Int, length(ind_set)) .- ind_set
    _set = findall(isone, ind_set)
    _comp = findall(isone, ind_comp)

    w_set = zeros(1, length(_set))
    w_comp = zeros(1, length(_comp))
    w_set .= idist_set' # first init
    wv = zeros(length(t_set) + length(t_comp) + 1)
    wv[1] = log(sum(idist_set))
    i = 2

    Q_h_comp = Q .* 1.0
    for ri in _comp
        Q_h_comp[ri, :] .= 0.0
    end
    UDU_comp = eigen(Q_h_comp)
    W_comp = UDU_comp.vectors
    W_comp_inv = inv(W_comp)
    D_comp = Diagonal(UDU_comp.values)
    aux_comp = zeros(size(D_comp)...)
    P_comp = zeros(size(Q)...)

    Q_h_set = Q .* 1.0
    for ri in _set
        Q_h_set[ri, :] .= 0.0
    end
    UDU_set = eigen(Q_h_set)
    W_set = UDU_set.vectors
    W_set_inv = inv(W_set)
    D_set = Diagonal(UDU_set.values)
    aux_set = zeros(size(D_comp)...)
    P_set = zeros(size(Q)...)

    R = zeros(size(Q)...)
    S = zeros(size(Q)...)
    _R = zeros(eltype(Q), length(_set), length(_comp))
    _S = zeros(eltype(Q), length(_comp), length(_set))

    for (ts, tc) in zip(t_set, t_comp)
        mul!(aux_comp, W_comp, exp(ts * D_comp))
        mul!(P_comp, aux_comp, W_comp_inv)
        mul!(R, Q_h_comp, P_comp)

        mul!(aux_set, W_set, exp(tc * D_set))
        mul!(P_set, aux_set, W_set_inv)
        mul!(S, Q_h_set, P_set)
        
        _select!(R, _R, _set, _comp)
        _select!(S, _S, _comp, _set)

        w_set ./= sum(w_set)
        mul!(w_comp, w_set, _R)
        wv[i] = sum(w_comp); i+=1

        w_comp ./= sum(w_comp)
        mul!(w_set, w_comp, _S)
        wv[i] = sum(w_set); i+=1
    end
    wv
    
end

let 
    date = 20241104
    c = 5.6
    kb = 0.0888
    kon = 5
    ku = 5
    koff = 5
    n = 1500
    k = 100
    dt = (1/(c*kb) + 1/kon + 1/ku + 1/koff)

    global m = LLROptModel(;
        weighted_edges = [(1,3,kb), 
                          (1,2,1e-5), 
                          (2,4,1e-5), 
                          (2,1,koff), 
                          (3,1,1e-5), 
                          (3,4,kon), 
                          (4,2,ku), 
                          (4,3,1e-5)],
        input_edges = [(1,3), (2,4)],
        on_ind = BitArray([0,1,0,1]),
        c = c,
        dc = c / 10,
        n = n,
        k = k,
        idist = [0, 0.5, 0, 0.5]
    )
    init!(m)

    l_vals = Vector{Float64}(undef, k)
    ll_vals = Vector{Float64}(undef, k)
    
    for i in 1:k
        generate!(m)
        get_residence_time!(m)
        m.ind_set = m.s_in_on ? m.on_ind : m.off_ind
        m.t_set = m.s_in_on ? m.onrt : m.offrt
        m.t_comp = m.s_in_on ? m.offrt : m.onrt
        m.idist_set = m.s_in_on ? m._idist0_on : m._idist0_off
        lf = l_full(m.Q_c, m.ind_set, m.t_set, m.t_comp, m.idist_set)
        llf = ll_full(m.Q_c, m.ind_set, m.t_set, m.t_comp, m.idist_set)
        l_vals[i] = prod(lf[2:end-1])
        ll_vals[i] = sum(llf)
    end

    @show Statistics.mean(ll_vals)
    @show log(Statistics.mean(l_vals))  ## promising!
end