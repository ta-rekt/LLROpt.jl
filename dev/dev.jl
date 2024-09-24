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
    n = 100000
    k = 20000
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

function ll_full_rmc(Q, ind_set, t_set, t_comp, idist_set)
    # @assert length(t_set) == length(t_comp)
    ind_comp = ones(Int, length(ind_set)) .- ind_set
    _set = findall(isone, ind_set)
    _comp = findall(isone, ind_comp)

    w_set = zeros(1, length(_set))
    w_comp = zeros(1, length(_comp))
    w_set .= idist_set' # first init
    wv = zeros(length(t_set) + length(t_comp) + 1)
    wv[1] = log(sum(idist_set))
    i = 2

    # Q_h_comp (fpt)
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
    exp_tD_comp = zeros(size(Q)...)

    # Q_h_set (fpt)
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
    exp_tD_set = zeros(size(Q)...)

    R = zeros(size(Q)...)
    S = zeros(size(Q)...)
    _R = zeros(eltype(Q), length(_set), length(_comp))
    _S = zeros(eltype(Q), length(_comp), length(_set))

    for (ts, tc) in zip(t_set, t_comp)
        # P = W * exp(t * D) * inv(W)
        # R = Q_h_comp * P(UDU_comp, ts)
        mul!(aux_comp, W_comp, exp(ts * D_comp))
        mul!(P_comp, aux_comp, W_comp_inv)
        mul!(R, Q_h_comp, P_comp)

        # P = W * exp(t * D) * inv(W)
        # S = Q_h_set * P(UDU_set, tc)
        
        # mul!(exp_tD_set, tc, D_set)
        # exp!(exp_tD_set)
        # mul!(aux_set, W_set, exp_tD_set)
        # mul!(aux_set, W_set, exp(exp_tD_set))

        mul!(aux_set, W_set, exp(tc * D_set))
        mul!(P_set, aux_set, W_set_inv)
        mul!(S, Q_h_set, P_set)
        
        _select!(R, _R, _set, _comp)
        _select!(S, _S, _comp, _set)

        # TODO: do it inline
        w_set ./= sum(w_set)
        # w_comp = (w_set' * _R)'
        mul!(w_comp, w_set, _R)
        wv[i] = log(sum(w_comp)); i+=1

        w_comp ./= sum(w_comp)
        mul!(w_set, w_comp, _S)
        # w_set = ((w_comp ./ sum(w_comp))' * _S)'
        wv[i] = log(sum(w_set)); i+=1
    end
    wv
end

function llr_rmc!(m::LLROptModel)
    # select based on m.s_in_on
    m.ind_set = m.s_in_on ? m.on_ind : m.off_ind
    m.t_set = m.s_in_on ? m.onrt : m.offrt
    m.t_comp = m.s_in_on ? m.offrt : m.onrt
    m.idist_set = m.s_in_on ? m._idist0_on : m._idist0_off
    
    # TODO: inline
    m.ll_c = ll_full_rmc(m.Q_c, m.ind_set, m.t_set, m.t_comp, m.idist_set)
    m.ll_cdc = ll_full_rmc(m.Q_cdc, m.ind_set, m.t_set, m.t_comp, m.idist_set)
    m.llr = cumsum(m.ll_c .- m.ll_cdc)
    return m
end

let 
    

end
