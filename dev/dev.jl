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
end


## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing estimators for the mean of gaussian increments with equal mean and variance
# dX = Vdt + sqrt(2Vdt)*eta_t

@time let
    date = 20240822
    nruns = 10000
    est1 = Vector{Float32}(undef, nruns)
    est2 = Vector{Float32}(undef, nruns)
    est3 = Vector{Float32}(undef, nruns)
    c = 5.6
    kb = 0.0888
    kon = 5
    ku = 5
    koff = 5
    n = 1000000
    k = 10000
    dt = (1/(c*kb) + 1/kon + 1/ku + 1/koff) * (n/k) / 2
    mu = 0.0017
    e = Normal(mu*dt, sqrt(2*mu*dt))

    for i in 1:nruns
        dx = rand(e, k)
        x = sum(dx)
        est1[i] = x / (dt*k)        # estimator 1: empirical mean
        est2[i] = var(dx) / (2*dt)  # estimator 2: empirical variance
        map!(y -> y^2, dx, dx)
        est3[i] = (-1 + sqrt(1 + sum(dx))) / (sqrt(k)*dt)      # estimator 3: MLE given V=D
    end

    path = "data/$(date)/"
    Base.Filesystem.mkpath(path)
    file1 = string(path, "empirical_mean_estimator.json")
    file2 = string(path, "empirical_variance_estimator.json")
    file3 = string(path, "maximum_likelihood_estimator.json")

    open(file1, "w") do f
        JSON.print(f, est1, 4)
    end
    open(file2, "w") do f
        JSON.print(f, est2, 4)
    end
    open(file3, "w") do f
        JSON.print(f, est3, 4)
    end
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: How gaussian are LLR increments?

@time let
    date = 20240822
    c = 5.6
    kb = 0.0888
    kon = 5
    ku = 5
    koff = 5
    n = 1000000
    k = 500000
    dt = (1/(c*kb) + 1/kon + 1/ku + 1/koff) * (n/k) / 2

    llr_incrs = Vector{Float32}(undef, n)
    llr_incrs_sub = Vector{Float32}(undef, k)

    global m0 = LLROptModel(;
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
        idist = ones(4) ./ 4,
    )
    init!(m0)
    UDU = eigen(m0.Q_c)
    dLLRs = Vector{Float32}(undef, n)
    dXs = Vector{Float32}(undef, k)
    nskip = Int(n/k)

    ir_full!(m0)
    llr_incrs = diff(m0.llr)
    llr_incrs_sub = diff(m0.llr[1:nskip:end])
    
    path = "data/$(date)/"
    Base.Filesystem.mkpath(path)
    file1 = string(path, "llr_incrs.json")
    file2 = string(path, "llr_incrs_sub.json")

    open(file1, "w") do f
        JSON.print(f, llr_incrs, 4)
    end

    open(file2, "w") do f
        JSON.print(f, llr_incrs_sub, 4)
    end

    nothing
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing mean and variance estimators with our LLR process

@time let

    date = 20240913
    c = 5.6
    kb = 0.0888
    kon = 5
    ku = 5
    koff = 5
    n = 10000
    k = 2000
    nruns = 1000
    dt = (1/(c*kb) + 1/kon + 1/ku + 1/koff) * (n/k) / 2

    ir_full_mean = Vector{Float32}(undef, nruns)
    ir_full_var = Vector{Float32}(undef, nruns)

    global m0 = LLROptModel(;
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
        idist = ones(4) ./ 4,
        k = 1
    )
    init!(m0)
    UDU = eigen(m0.Q_c)
    dXs = Vector{Float32}(undef, k)
    nskip = Int(n/k)

    for i in 1:nruns
        ir_full!(m0)
        dXs = diff(m0.llr[1:nskip:end-1])
        time = 2 .* cumsum(m0.rt)
        sq_deltat = map(x->sqrt(x), diff(time[1:nskip:end]))
        ir_full_mean[i] = m0.llr[end] / sum(m0.rt)
        ir_full_var[i] = var(dXs ./ sq_deltat)
    end

    path = "data/$(date)/"
    Base.Filesystem.mkpath(path)
    file1 = string(path, "ir_full_mean_estimator.json")
    file2 = string(path, "ir_full_variance_estimator.json")

    open(file1, "w") do f
        JSON.print(f, ir_full_mean, 4)
    end

    open(file2, "w") do f
        JSON.print(f, ir_full_var, 4)
    end

    nothing
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing text IO

let 
    file = "test.txt"

    open(file, "w") do f
        for i in 1:5
            line = string(i, i+1)
            
            println(f, line)
        end
    end
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing JSON
using JSON

let 
    Random.seed!(123)
    nrow = 3
    ncol = 2
    data = rand(nrow,ncol)
    dict = Dict(i => data[i:nrow:nrow*ncol] for i = 1:nrow)

    date = 20240730
    key = 1
    
    path = "data/$(date)/"
    Base.Filesystem.mkpath(path)
    file = string(path, "test$(key).json")

    open(file, "w") do f
        JSON.print(f, dict, 4)
    end
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing P_ss

let 
    date = 20240911
    c = 5.6
    a = 3e-3        # um
    D_tf = 7.4      # um^2/s
    D_cplx = 0.1    # um^2/s
    w = 0.2
    cr = 5.6         # um^-3
    dc = 0.1 * cr
    cw = w * cr
    gamma = 1
    n = 100000
    k = 20000
    nruns = 1000
    # dt = (1/(c*kb) + 1/kon + 1/ku + 1/koff) * (n/k) / 2

    kboff = 3
    konu = 1
    kbon = 1
    koffu = 5
    kuoff = 1
    konb = 5
    kuon = 5
    koffb = 1

    wedges = [(5,1,gamma*kuoff),
             (1,5,cw*kboff),
             (2,6,cw*kbon),
             (6,2,gamma*kuon),
             (1,2,konu), 
             (1,3,kboff), 
             (2,4,kbon), 
             (2,1,koffu), 
             (3,1,kuoff), 
             (3,4,konb), 
             (4,2,kuon), 
             (4,3,koffb),
             (5,6,konb),
             (6,5,koffb)]

    global m0 = LLROptModel(;
        weighted_edges = wedges,
        input_edges = [(1,3), (2,4)],
        on_ind = BitArray([0,1,0,1,0,1]),
        c = c,
        dc = c / 10,
        n = n,
        idist = ones(6) ./ 6,
    )
    init!(m0)
    Pi = P_ss(m0.Q_c)
    log10(Pi[4]/Pi[6])
end 

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: implementing the RMC algorithm from J. Vanneste PHYSICAL REVIEW E 81, 036701 (2010)

function generate_rmc!(m::LLROptModel)

    m.s, m.t = zeros(Int, m.k, m.n), zeros(Float64, m.k, m.n)
    _DD = _ddist(m.Q_c)

    @assert size(_DD) == size(m.Q_c)

    for j in 1:m.k
        m.s[j,1] = sample(eachindex(m.idist), Weights(m.idist))
    end
    
    for i in 1:m.n-1
        for j in 1:m.k
            
            si = m.s[j,i]
            q = -m.Q_c[si, si]

            # t
            e = Exponential(1/q)
            m.t[j,i] = rand(e)

            # s        
            _dd = _DD[si, :]
            m.s[j,i+1] = sample(eachindex(_dd), Weights(_dd))
        end
    end

    for j in 1:m.k
        # t
        si = m.s[j,end]
        q = -m.Q_c[si, si]
        e = Exponential(1/q)
        m.t[j,end] = rand(e)
    end
end

function get_residence_time_rmc!(m::LLROptModel)

    m.rt = zeros(Float64, m.k, m.n)    # residence times
    m.t_set = zeros(Float64, m.k, m.n)  # set residence times
    m.t_comp = zeros(Float64, m.k, m.n) # complement set residence times
    m.s_in_on = zeros(Int, m.k)        # start in on indicator
    m.jc = zeros(Int, m.k)             # jump counts
    m.bc = zeros(Int, m.k)             # burst counts
    
    # loop across trajectories
    for j in 1:m.k
        sj = view(m.s, j, :)
        tj = view(m.t, j, :)
        t_setj = view(m.t_set, j, :)
        t_compj = view(m.t_comp, j, :)
        jc = _jump_count(sj, m.on_ind)
        bc = floor(Int32, jc / 2)    # burst count

        _last_is_on = m.on_ind[sj[1]]
        i_jump = 1 # jump index

        # loop across timesteps of each trajectory
        for (sij, tij) in zip(sj, tj)
            if (m.on_ind[sij] ⊻ _last_is_on)
                _last_is_on = m.on_ind[sij]
                i_jump += 1
            end
            m.rt[j, i_jump] += tij
        end

        # postprocess
        m.jc[j] = jc
        m.bc[j] = bc
        m.s_in_on[j] = m.on_ind[sj[1]]
        t_setj[1:bc] = m.rt[j, 1:2:2*bc]
        t_compj[1:bc] = m.rt[j, 2:2:2*bc+1]
    end
end

# assumes all trajectories start in the same partition
function ll_full_rmc(Q, on_ind, idist, bc, t_set, t_comp)

    @assert length(t_set) == length(t_comp)

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

    min_bc = minimum(bc)  # length of shortest trajectory
    n = min_bc*2 + 1
    k = size(t_set)[1]   # number of trajectories

    w_on = zeros(1, length(_on))
    w_off = zeros(1, length(_off))
    w_on .= _idist_on'   # first init
    wv = zeros(Float64, k, n)    # log vector norms
    vec = zeros(Float64, k, length(w_on))
    vec_sampled = zeros(Float64, k, length(w_on))
    alpha = zeros(Float64, k)
    logbeta = zeros(Float64, min_bc)
    t_on, t_off = 0.0, 0.0

    #### trajectory-dependent operations ####

    for j in 1:k
        wv[j,1] = sum(w_on)
        vec[j,:] = w_on
    end

    for i in 1:min_bc
        jmp = 2
        betasum = 0
        jsampl = 0

        for j in 1:k
            t_on, t_off = t_set[j,i], t_comp[j,i]

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
            w_on ./= alpha[j]
            mul!(w_off, w_on, _R)
            wv[j,jmp] = log(sum(w_off)); jmp += 1

            w_off ./= sum(w_off)
            mul!(w_on, w_off, _S)
            wv[j,jmp] = log(sum(w_on)); jmp += 1
            vec[j,:] = w_on'

            betasum += alpha[j]
        end

        alpha ./= betasum
        logbeta[i] = log(betasum)

        for j in 1:k
            jsampl = sample(eachindex(alpha), Weights(alpha))
            vec_sampled[j,:] = vec[jsampl,:]
        end
    end

    # return log sum of betas - logK
    sum(logbeta) / min_bc - log(k)
end

function llr_rmc!(m::LLROptModel)
    # TODO: inline
    m.ll_c = ll_full_rmc(m.Q_c, m.on_ind, m.idist, m.bc, m.t_set, m.t_comp)
    m.ll_cdc = ll_full_rmc(m.Q_cdc, m.on_ind, m.idist, m.bc, m.t_set, m.t_comp)
    m.llr = m.ll_c .- m.ll_cdc
    return m
end

function ir_full_rmc!(m::LLROptModel)
    generate_rmc!(m)
    get_residence_time_rmc!(m)
    llr_rmc!(m::LLROptModel)
end

@time let 
    date = 20240913
    c = 5.6
    kb = 0.0888
    kon = 5
    ku = 5
    koff = 5
    n = 10
    k = 5
    dt = (1/(c*kb) + 1/kon + 1/ku + 1/koff)

    global m = LLROptModel(;
        weighted_edges = [(1,3,kb), 
                          (1,2,1e0), 
                          (2,4,1e0), 
                          (2,1,koff), 
                          (3,1,1e0), 
                          (3,4,kon), 
                          (4,2,ku), 
                          (4,3,1e0)],
        input_edges = [(1,3), (2,4)],
        on_ind = BitArray([0,1,0,1]),
        c = c,
        dc = c / 10,
        n = n,
        k = k,
        idist = [0, 0.5, 0, 0.5]
    )
    init!(m)
    generate_rmc!(m)
    get_residence_time_rmc!(m)
    ll_full_rmc(m.Q_c, m.on_ind, m.idist, m.bc, m.t_set, m.t_comp)
end
