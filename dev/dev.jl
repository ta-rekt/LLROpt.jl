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
    dirname = 20240822
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

    path = "data/$(dirname)/"
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
    dirname = 20240822
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
    
    path = "data/$(dirname)/"
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

    dirname = 20241009
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

    path = "data/$(dirname)/"
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
# DOING: testing V=D across time

@time let

    dirname = "20241102/max_IR_equilibrium_circuit"
    c = 5.6

    ### MGR circuit
    # kboff = 0.0888
    # kbon = 1e-5
    # konu = 1e-5
    # konb = 5
    # kuon = 5
    # kuoff = 1e-5
    # koffu = 5
    # koffb = 1e-5
    # taub = (1/(c*kboff) + 1/konb + 1/kuon + 1/koffu)

    ### max-IRacc circuit (with kuoff > 0)
    # kboff = 0.0888
    # kbon = 1e-5
    # konu = 1e-5
    # konb = 5
    # kuon = 5
    # kuoff = 5
    # koffu = 5
    # taub = 10
    # koffb = 1e-5

    ### max-IR equilibrium circuit
    kboff = 0.0888
    kbon = 1e-5
    konu = 1e-5
    konb = 5
    kuon = 1e-5
    kuoff = 5
    koffu = 1e-5
    koffb = 5

    n = 5000
    nruns = 50000
    ntimes = 200
    nerrs = 100
    errs = range(0.5, stop=1e-2, length=nerrs)

    #### LLR process

    ir_full_samples = Matrix{Float32}(undef, nruns, ntimes)
    stop_times_llr = Matrix{Float32}(undef, nruns, nerrs)
    stop_times_gp = Matrix{Float32}(undef, nruns, nerrs)

    global m0 = LLROptModel(;
        weighted_edges = [(1,3,kboff), 
                          (1,2,konu), 
                          (2,4,kbon), 
                          (2,1,koffu), 
                          (3,1,kuoff), 
                          (3,4,konb), 
                          (4,2,kuon), 
                          (4,3,koffb)],
        input_edges = [(1,3), (2,4)],
        on_ind = BitArray([0,1,0,1]),
        c = c,
        dc = c / 10,
        n = n,
        idist = ones(4) ./ 4,
        k = 1
    )
    init!(m0)
    mu = 0.001728
    nsteps = 5000
    dt = 1e-1
    e = Normal(mu*dt, sqrt(2*mu*dt))

    for i in 1:nruns

        ### LLR process
        ir_full!(m0)
        time = cumsum(m0.rt)
        times = range(1, stop=time[end]/2, length=ntimes)

        ### True gaussian process with V=D
        dx = rand(e, nsteps)
        xt = cumsum(dx)

        for (j,t) in enumerate(times)
            k = findfirst(x -> x > t, time)

            if k === nothing
                ir_full_samples[i, j] = m0.llr[end] / t
            else
                ir_full_samples[i, j] = m0.llr[k] / t
            end
        end

        for (p,e) in enumerate(errs)
            K = log((1-e)/e)
            q1 = findfirst(x -> x > K || x < (-K), m0.llr[2:end])
            q2 = findfirst(x -> x > K || x < (-K), xt)

            if q1 === nothing
                stop_times_llr[i, p] = time[end]
            else
                stop_times_llr[i, p] = time[q1]
            end

            if q2 === nothing
                stop_times_gp[i, p] = time[end]
            else
                stop_times_gp[i, p] = q2*dt
            end
        end
    end

    path = "data/$(dirname)/"
    Base.Filesystem.mkpath(path)
    file1 = string(path, "ir_full_samples.json")
    file2 = string(path, "stop_times_llr.json")
    file3 = string(path, "stop_times_gp.json")

    open(file1, "w") do f
        JSON.print(f, ir_full_samples, 4)
    end

    open(file2, "w") do f
        JSON.print(f, stop_times_llr, 4)
    end

    open(file3, "w") do f
        JSON.print(f, stop_times_gp, 4)
    end

    nothing
end


## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: Improving decision time statistics ?

let 
    mu = 0.001728
    n = 50000
    dt = 0.1
    e = Normal(mu*dt, sqrt(2*mu*dt))
    global dxs = rand(e, n)
    global acc = 0.0
    er = 0.35
    global K = log((1.0-er)/er)
    ub, lb = -K, K

    global starts = Vector{Float32}(undef, round(Int, n/2))
    global ends = Vector{Float32}(undef, round(Int, n/2))
    ji = 1
    starts[ji] = 1

    for (i,dx) in enumerate(dxs)
        acc += dx

        if acc > K || acc < (-K)
            ends[ji] = i
            starts[ji + 1] = i + 1
            acc = 0
            ji += 1
        end
    end

    times = Vector{Float32}(undef, ji - 1)

    for j in 1:(ji - 1)
        times[j] = (ends[j] - starts[j]) * dt
    end

    @show times

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

    dirname = 20240730
    key = 1
    
    path = "data/$(dirname)/"
    Base.Filesystem.mkpath(path)
    file = string(path, "test$(key).json")

    open(file, "w") do f
        JSON.print(f, dict, 4)
    end
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing P_ss

let 
    dirname = 20240911
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
