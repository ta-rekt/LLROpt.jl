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
    using QuadGK
    using Cubature
    using DoubleExponentialFormulas
    using Plots
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

    # dirname = 20241103
    # dirname = 20241111
    dirname = 20241125
    c = 5.6
    kboff = 0.0888
    konu = 1e-5
    kbon = 0.0888
    koffu = 1e-5
    kuoff = 5
    konb = 5
    kuon = 5
    koffb = 5
    n = 1000000
    ntrajs = 1
    nrepeats = 100
    # taub = (1/(c*kb) + 1/kon + 1/ku + 1/koff)
    # tau = 60*taub

    ir_full_mean = Vector{Float64}(undef, nrepeats)
    ir_full_var = Vector{Float64}(undef, nrepeats)
    samples = Vector{Float64}(undef, ntrajs)

    global m1 = LLROptModel(;
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
    # global m = LLROptModel(;
    # weighted_edges = [(5,1, kb),  # kb
    #                   (1,5, 1e-5), 
    #                   (2,6, ku),    # ku
    #                   (6,2, 1e-5), 
    #                   (1,2, kon),   # kon
    #                   (1,3, kb),    # kb 
    #                   (2,4, 1e-5),
    #                   (2,1, 1e-5), 
    #                   (3,1, 1e-5), 
    #                   (3,4, 2*kon),   # kon
    #                   (4,2, 2*ku),  # ku
    #                   (4,3, 1e-5), 
    #                   (5,6, 1e-5), 
    #                   (6,5, koff)], # koff
    #     input_edges = [(1,3), (2,4), (5,1), (6,2)],
    #     on_ind = BitArray([0,1,0,1,0,1]),
    #     c = c,
    #     dc = c / 10.0,
    #     n = n,
    #     idist = ones(6) ./ 6,
    #     k = 1
    # )
    # init!(m)
    init!(m1)

    for i in 1:nrepeats
        for j in 1:ntrajs
            ir_full!(m1)
            time = cumsum(m1.rt)
            # k = findfirst(x -> x > tau, time)
            # samples[j] = m.llr[k] / tau
            samples[j] = m1.llr[end] / time[end]
        end

        ir_full_mean[i] = mean(samples)
        # ir_full_var[i] = var(samples .* sqrt(tau)) / 2
    end

    path = "data/$(dirname)/"
    Base.Filesystem.mkpath(path)
    file1 = string(path, "ir_full_mean_estimator.json")
    # file2 = string(path, "ir_full_variance_estimator.json")

    open(file1, "w") do f
        JSON.print(f, ir_full_mean, 4)
    end

    # open(file2, "w") do f
    #     JSON.print(f, ir_full_var, 4)
    # end

    @show Statistics.mean(ir_full_mean)
    @show sqrt(Statistics.var(ir_full_mean))
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing V=D and decision time distribution

@time let

    dirname = "20241125"
    c = 5.6

    ### MGR circuit
    kboff = 0.0888
    kbon = 1e-5
    konu = 1e-5
    konb = 5
    kuon = 5
    kuoff = 1e-5
    koffu = 5
    koffb = 1e-5
    taub = (1/(c*kboff) + 1/konb + 1/kuon + 1/koffu)

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
    # kboff = 0.0888
    # kbon = 1e-5
    # konu = 1e-5
    # konb = 5
    # kuon = 1e-5
    # kuoff = 5
    # koffu = 1e-5
    # koffb = 5

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
        times = range(taub, stop=n*taub/8, length=ntimes)

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
    dirname = "20241109/test-st"

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

    path = "data/$(dirname)/"
    Base.Filesystem.mkpath(path)
    file2 = string(path, "stop_times_llr.json")

    open(file1, "w") do f
        JSON.print(f, ir_full_samples, 4)
    end

end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: Testing decorrelated activation times idea

@time let

    dirname = "20241109/test-decorr-act-times"
    c = 5.6

    ### MGR circuit
    kboff = 0.0888
    kbon = 1e-5
    konu = 1e-5
    konb = 5
    kuon = 5
    kuoff = 1e-5
    koffu = 5
    koffb = 1e-5
    taub = (1/(c*kboff) + 1/konb + 1/kuon + 1/koffu)

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
    # kboff = 0.0888
    # kbon = 1e-5
    # konu = 1e-5
    # konb = 5
    # kuon = 1e-5
    # kuoff = 5
    # koffu = 1e-5
    # koffb = 5

    n = 5000
    nruns = 50000

    #### Estimating D(p||q) where p is activity likelihood, q binding likelihood

    dkl_values = Matrix{Float32}(undef, nruns, ncircuits)

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

    for i in 1:nruns
        # get residence times of binding times
        # use ll_full just like with activation times
        
    end

    path = "data/$(dirname)/"
    Base.Filesystem.mkpath(path)
    file1 = string(path, "dkl_values.json")

    open(file1, "w") do f
        JSON.print(f, dkl_values, 4)
    end

    nothing
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: Sample & project algorithm for walking along a pareto front

let 
    import Pkg
    Pkg.activate("LLROpt")
    using JSON
    using Statistics
    using LLROpt
end

@time let 

    specfile = "1bs_sens_spec.jl"
    cpukey = 1
    ncpus = 1
    outdir = "20241123/test-point-cloud"

    # output directory
    outpath = "data/$(outdir)/"
    Base.Filesystem.mkpath(outpath)

    # specification file
    specpath = string("../spec/", specfile)
    include(specpath)

    # input data
    if inpath !== ""
        infile = string(inpath, cpukey, ".json")
        results_init = Dict()
        open(infile, "r") do f
            results_init = JSON.parse(f)
        end
    end

    ### Bounds
    llb = llb1
    lub = lub1 .+ 1 .* ones(8)

    ### MGR circuit
    kboff = log10(0.0888)
    kbon = log10(0.0888)
    konu = -5
    konb = log10(5)
    kuon = log10(5)
    kuoff = -5
    koffu = log10(5)
    koffb = -5
    beta = 0
    keyb = 1

    npoints = 100
    betas = zeros(npoints)
    nparams = length(betas)

    xinit1 = [kboff, konu, kbon, koffu, kuoff, konb, kuon, koffb]
    logcw = log10(cr) 
    kboff, konu, kbon, koffu, kuoff, konb, kuon, koffb = xinit1
    xinit2 = [kuoff, kboff+logcw, kbon+logcw, kuon, 
        konu, kboff, kbon, koffu, kuoff, konb, kuon, koffb, konb, koffb]

    nedges1 = 8
    nedges2 = 14

    sig = 0.01
    project = true

    irvals_mix = Vector{Float32}(undef, N)
    irvals_pure = Vector{Float32}(undef, N)
    results_full = Dict{Int64, Dict{Int64, Dict{Int64, Vector{Float32}}}}()
    xs = Matrix{Float64}(undef, npoints, nedges1 + 4)  # rates, ir, dr, ep, eps
    x = Vector{Float64}(undef, nedges1)

    we1 = [(u,v,r) for ((u,v),r) in zip(edges1, xinit1)]
    we2 = [(u,v,r) for ((u,v),r) in zip(edges2, xinit2)]

    m1 = LLROptModel(;
        weighted_edges = we1,
        input_edges = input_edges,
        on_ind = on_ind1,
        c = cr,
        dc = dc,
        n = n,
        k = k,
        idist = idist1,
    )

    m = LLROptModel(;
        weighted_edges = we2,
        input_edges = input_edges,
        on_ind = on_ind2,
        c = cr,
        dc = dc,
        n = n,
        k = k,
        idist = idist2,
    )

    init!(m1)
    init!(m)

    # generate some random circuits around MGR
    for keyb in 1:npoints
        results_full[keyb] = Dict{Int64, Dict{Int64, Vector{Float32}}}()

        for (keyg, gamma) in enumerate(gammas)
            results_full[keyb][keyg] = Dict{Int64, Vector{Float32}}()

            for (keyw, w) in enumerate(wvals)
                if inpath !== ""
                    xinit = results_init[string(keyb)][string(keyg)][string(keyw)][1:end-8]
                    xinit = Vector{Float64}(xinit)
                else
                    xinit = xinit1
                end

                for i in 1:nedges1
                    e = Normal(xinit1[i], sig)

                    if keyb == 1
                        x[i] = xinit1[i]
                    else
                        x[i] = rand(truncated(e, llb[i], lub[i]))
                    end
                    xs[keyb,i] = x[i]
                end

                args_full = [m1, m, beta, gamma, w]
                targetfun_full(x_, p) = target_full(x_, args_full...)[1]
                ir_full_mix(x_) = target_full(x_, args_full...)[2]
                ir_full_pure(x_) = target_full(x_, args_full...)[3]
                ep_full(x_) = target_full(x_, args_full...)[4]

                # RESULTS
                if project == true
                    for i in 1:nedges1
                        if x[i] > lub1[i]
                            x[i] = lub1[i]
                        elseif x[i] < llb1[i]
                            x[i] = llb1[i]
                        end
                    end
                end

                # if inbounds == true
                    results_full[keyb][keyg][keyw] = vcat(x, 0, 0, 0, 0, 
                        ep_full(x), beta, gamma, w)
            
                    for i in 1:N
                        irvals_mix[i] = ir_full_mix(x)
                        irvals_pure[i] = ir_full_pure(x)
                    end

                    results_full[keyb][keyg][keyw][end-7] = Statistics.mean(irvals_mix)
                    results_full[keyb][keyg][keyw][end-6] = Statistics.mean(irvals_pure)
                    results_full[keyb][keyg][keyw][end-5] = Statistics.std(irvals_mix)
                    results_full[keyb][keyg][keyw][end-4] = Statistics.std(irvals_pure)
                # else
                #     results_full[keyb][keyg][keyw] = zeros(16)
                # end
            end
        end
    end

    # minf = 1e6
    # thisf = 1e6
    # minx = zeros(8)

    # # put the points loop inside and keep the betas loop outside
    # # for each beta, select the point that minimizes the target for the next beta
    # # keep sampling until your point is within epsilon of the minimum (optional)

    # for (keyg, gamma) in enumerate(gammas)
    #     for (keyw, w) in enumerate(wvals)
    #         for keyb in 1:npoints
    #             x = results_full[keyb][keyg][keyw][1:8]
    #             thisf = targetfun_full(x)

    #             if thisf < minf
    #                 minf = thisf
    #                 minx = x
    #             end
    #         end
    #     end
    # end

    file2 = string(outpath, "ir_full_mix_vs_pure_pareto_pid$(cpukey).json")

    open(file2, "w") do f
        JSON.print(f, results_full, 4)
    end
    params = Dict{String, Any}()
    params["ncpus"] = ncpus
    params["nparams"] = nparams
    params["betas"] = betas
    params["gammas"] = gammas
    params["wvals"] = wvals
    params["n"] = n
    params["N"] = N
    params["maxiters_acc"] = maxiters_acc
    params["maxiters_full"] = maxiters_full
    params["ncalls_full"] = COUNTERS["target_full"]

    paramfile = string(outpath, "params.json")
        
    open(paramfile, "w") do f
        JSON.print(f, params, 4)
    end
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: Is there symmetry-breaking with 2 binding sites? yes there is

let 
    specfile = "2bs_sens_epr.jl"
    cpukey = 1
    ncpus = 1

    # specification file
    specpath = string("../spec/", specfile)
    include(specpath)

    m = LLROptModel(;
        weighted_edges = we,
        input_edges = input_edges,
        on_ind = on_ind,
        c = c,
        dc = dc,
        n = n,
        k = k,
        idist = idist,
    )

    edges = [(5,1), (1,5), (2,6), (6,2), (1,2), (1,3), (2,4),
             (2,1), (3,1), (3,4), (4,2), (4,3), (5,6), (6,5)]

    init!(m)
    beta = 0
    e = (-1,0)
    args_acc = [edges, m.input_edges, c, dc, on_ind, beta, e]

    logrates = [-1.0515876,
                -3,
                -1.0515876,
                0.69896996,
                -2.9974964,
                0.69896996,
                0.6989698,
                -2.9998689,
                0.69896996,
                -2.9999237]

    # logrates = [-1.0515876,
    #             0.6989699,
    #             -1.0515876,
    #             -2.9999237,
    #             0.6989698,
    #             -2.999996,
    #             -2.9974964,
    #             -2.9998689,
    #             -2.9996033,
    #             0.69896996]

    targ, ir, ep = ir_acc_log(logrates, args_acc...)

end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: How close are the V upper and lower bounds?

function ll_lb(Q, ind_set, t_set, t_comp, idist_set)
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

    d.Q_h_comp = Q .* 1.0
    for ri in _comp
        d.Q_h_comp[ri, :] .= 0.0
    end
    UDU_comp = eigen(d.Q_h_comp)
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
        mul!(R, d.Q_h_comp, P_comp)

        mul!(aux_set, W_set, exp(tc * D_set))
        mul!(P_set, aux_set, W_set_inv)
        mul!(S, Q_h_set, P_set)
        
        _select!(R, _R, _set, _comp)
        _select!(S, _S, _comp, _set)

        w_set ./= sum(w_set)
        mul!(w_comp, w_set, _R)
        wv[i] = log(sum(w_comp)); i+=1

        w_comp ./= sum(w_comp)
        mul!(w_set, w_comp, _S)
        wv[i] = log(sum(w_set)); i+=1
    end
    wv
    
end

@time let

    dirname = 20241125
    c = 5.6
    kboff = 0.0888
    konu = 1e-5
    kbon = 0.0888
    koffu = 1e-5
    kuoff = 5
    konb = 5
    kuon = 5
    koffb = 5
    n = 1000000
    ntrajs = 1
    nrepeats = 100

    ir_full_mean = Vector{Float64}(undef, nrepeats)
    ir_full_var = Vector{Float64}(undef, nrepeats)
    samples = Vector{Float64}(undef, ntrajs)

    global m = LLROptModel(;
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

    init!(m)

    for i in 1:nrepeats
        for j in 1:ntrajs
            generate!(m)
            get_residence_time!(m)

            m.ind_set = m.s_in_on ? m.on_ind : m.off_ind
            m.t_set = m.s_in_on ? m.onrt : m.offrt
            m.t_comp = m.s_in_on ? m.offrt : m.onrt
            m.idist_set = m.s_in_on ? m._idist0_on : m._idist0_off

            ll_lb_c = ll_lb(m.Q_c, m.ind_set, m.t_set, m.t_comp, m.idist_set)
            ll_lb_cdc = ll_lb(m.Q_cdc, m.ind_set, m.t_set, m.t_comp, m.idist_set)

            ll_c = ll_full(m.Q_c, m.ind_set, m.t_set, m.t_comp, m.idist_set)
            ll_cdc = ll_full(m.Q_cdc, m.ind_set, m.t_set, m.t_comp, m.idist_set)

            time = cumsum(m.rt)
            samples[j] = m.llr[end] / time[end]
        end

        ir_full_mean[i] = mean(samples)
    end

    path = "data/$(dirname)/"
    Base.Filesystem.mkpath(path)
    file1 = string(path, "ir_full_mean_estimator.json")

    open(file1, "w") do f
        JSON.print(f, ir_full_mean, 4)
    end

    @show Statistics.mean(ir_full_mean)
    @show sqrt(Statistics.var(ir_full_mean))
end


## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: testing JSON
using JSON
using Random

let 
    Random.seed!(123)
    nrow = 3
    ncol = 6
    data = rand(nrow, ncol)
    dict = Dict(i => data[i:nrow:nrow*ncol] for i = 1:nrow)

    dirname = 20240730
    key = 1
    
    path = "data/$(dirname)/"
    # path = ""
    if path !== ""
        Base.Filesystem.mkpath(path)
    else
        @show "pimp"
    end
    file = string(path, "test$(key).json")

    open(file, "w") do f
        JSON.print(f, dict, 4)
    end
end

let 
    dirname = 20240730
    key = 1
    
    path = "data/$(dirname)/"
    Base.Filesystem.mkpath(path)
    file = string(path, "test$(key).json")
    s = Dict()

    open(file, "r") do f
        s = JSON.parse(f, dicttype=Dict{String, Vector{Float32}})
    end
    s["1"][1:end-3]
end

let
    inpath = "/home/tohme/.julia/dev/LLROpt/data/20241114/ir-epsilon-hernan/gen1/ir_acc_mix_vs_pure_pareto_pid"
    cpukey = 1
    infile = string(inpath, cpukey, ".json")
    results_init = Dict{String, Dict{String, Dict{String, Vector{Float32}}}}()
    open(infile, "r") do f
        results_init = JSON.parse(f)
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

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# DOING: exact integration of decoy circuit MLE

function tauOn(logrates, args...)
    cr, cw, gamma = args
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

function tauOff(logrates, args...)
    cr, cw, gamma = args
    omega = cw / cr
    kboff, konb, kuon, kbon, koffu, kuoff, konu = 10 .^ logrates

    numerator = (konb + kuoff)*(konb + kuoff*gamma) + cr*kboff*(konb*(1 + omega) + kuoff*(gamma + omega))
    denominator = (konu*(konb + kuoff)*(konb + kuoff*gamma) 
                   + cr*kboff*konb*(konb*(1 + omega) + kuoff*(gamma + omega)))
     
    return numerator / denominator

end


function initForward!(d::ForwardAlgModel)

    d.ind_comp = ones(Int, length(d.ind_set)) .- d.ind_set
    d._set = findall(isone, d.ind_set)
    d._comp = findall(isone, d.ind_comp)

    # d.Q_h_comp (fpt)
    d.Q_h_comp = d.Q .* 1.0
    for ri in d._comp
        d.Q_h_comp[ri, :] .= 0.0
    end
    UDU_comp = eigen(d.Q_h_comp)
    d.W_comp = UDU_comp.vectors
    d.W_comp_inv = inv(d.W_comp)
    d.D_comp = Diagonal(UDU_comp.values)
    d.aux_comp = zeros(size(d.D_comp)...)
    d.P_comp = zeros(size(d.Q)...)
    # exp_tD_comp = zeros(size(Q)...)

    # Q_h_set (fpt)
    d.Q_h_set = d.Q .* 1.0
    for ri in d._set
        d.Q_h_set[ri, :] .= 0.0
    end
    UDU_set = eigen(d.Q_h_set)
    d.W_set = UDU_set.vectors
    d.W_set_inv = inv(d.W_set)
    d.D_set = Diagonal(UDU_set.values)
    d.aux_set = zeros(size(d.D_comp)...)
    d.P_set = zeros(size(d.Q)...)
    # exp_tD_set = zeros(size(Q)...)

    d.F_set = zeros(size(d.Q)...)
    d.F_comp = zeros(size(d.Q)...)
    
    d.eigvals_set = UDU_set.values
    d.eigvals_comp = UDU_comp.values
    d.maxeigval_set = - maximum(x for x in d.eigvals_set if x < 0)
    d.maxeigval_comp = - maximum(x for x in d.eigvals_comp if x < 0)

    return d
end


function Pjoint!(t_off, t_on, d)

    s1 = d.set_start
    s3, s4, s5 = d.comp_arrive

    mul!(d.aux_comp, d.W_comp, exp(t_off * d.D_comp))
    mul!(d.P_comp, d.aux_comp, d.W_comp_inv)
    mul!(d.F_comp, d.Q_h_comp, d.P_comp)

    mul!(d.aux_set, d.W_set, exp(t_on * d.D_set))
    mul!(d.P_set, d.aux_set, d.W_set_inv)
    mul!(d.F_set, d.Q_h_set, d.P_set)

    d.Pjoint = max((d.F_comp[s1,s3] * d.F_set[s3,s1] 
                  + d.F_comp[s1,s4] * d.F_set[s4,s1] 
                  + d.F_comp[s1,s5] * d.F_set[s5,s1]), d.eps)

    return d
end

function Pjoint(t_off, t_on, d)

    Pjoint!(t_off, t_on, d)

    return d.Pjoint
end

function PjointVec(t, d)

    t_on, t_off = t

    return Pjoint(t_on, t_off, d)
end

function inner_integrand(t_off, t_on, d_c, d_cdc, m)

    Pjoint!(t_off, t_on, d_c)
    Pjoint!(t_off, t_on, d_cdc)

    return d_c.Pjoint * (log(d_c.Pjoint) - log(d_cdc.Pjoint))
end

function outer_integrand(t_on, d_c, d_cdc, m)
    integral, _ = quadgk(t -> inner_integrand(t, t_on, d_c, d_cdc, m), 0, 40/d_c.maxeigval_comp, rtol=1e-8)
    return integral
end


function target_mle_exact(logrates, args...)
    # Builds a decoys model and computes the expected LLR by numerical integration

    d_c, d_cdc, m, cr, gamma, cw = args
    tau = tauOn(logrates, cr, cw, gamma) + tauOff(logrates, cr, cw, gamma)

    m.c = cr
    m.dc = cr / 10

    kboff, konb, kuon, kbon, koffu, kuoff, konu = 10 .^ logrates

    ratesFull = [cr*kboff, kuoff,      cw*kboff, gamma*kuoff, cr*kbon, kuon,
                 cw*kbon,   gamma*kuon, konb,     koffu,       konb,     konu]

    for (keyb, val) in enumerate(ratesFull)
        (u,v,r) = m.weighted_edges[keyb]
        m.weighted_edges[keyb] = (u,v,val)
    end

    init!(m)

    d_c.Q = m.Q_c
    d_cdc.Q = m.Q_cdc
    d_c.ind_set = m.ind_set
    d_cdc.ind_set = m.ind_set

    initForward!(d_c)
    initForward!(d_cdc)

    Fonoff, _ = quadgk(t -> outer_integrand(t, d_c, d_cdc, m), 0, 20, rtol=1e-8)

    return 2 * Fonoff * (cr)^2 / (m.dc^2 * tau)
    # return nothing
end


function outer_integrand_test(t_on, d)

    lbound = 0
    neigs = length([x for x in d.eigvals_comp if x < 0])
    integral = 0

    for i in range(1, stop=neigs)
        if i == neigs
            rbound = -20/d_c.eigvals_set[i]
        else
            rbound = -5/d_c.eigvals_set[i]
        end
        piece, _ = quadde(t -> Pjoint(t, t_on, d), lbound, rbound, rtol=1e-10)
        integral = integral + piece
        lbound = rbound
    end

    return t_on*integral
end

function initForwardRates!(logrates, args)

    d_c, d_cdc, m, cr, gamma, cw = args

    kboff, konb, kuon, kbon, koffu, kuoff, konu = 10 .^ logrates

    ratesFull = [kboff,     kuoff,      cw*kboff, gamma*kuoff, kbon,     kuon,
                 cw*kbon,   gamma*kuon, konb,     koffu,       konb,     konu]

    for (keyb, val) in enumerate(ratesFull)
        (u,v,r) = m.weighted_edges[keyb]
        m.weighted_edges[keyb] = (u,v,val)
    end

    init!(m)  # which calls build_mat

    d_cdc.Q = m.Q_cdc
    d_c.Q = m.Q_c
    d_cdc.ind_set = m.ind_set
    d_c.ind_set = m.ind_set

    initForward!(d_c)
    initForward!(d_cdc)

end

@time let 
    edges = [(1, 2), (2, 1), (1, 6), (6, 1), (4, 3), (3, 4), 
             (4, 5), (5, 4), (6, 5), (4, 1), (2, 3), (1, 4)]
    input_edges = [(1,2), (4,3)]
    off_ind = BitArray([1,1,0,0,0,1])
    on_ind = BitArray([0,0,1,1,1,0])
    idist = ones(6) ./ 6
    n = 1
    k = 1
    cr = 1e4
    dc = cr / 10
    beta = cr
    gamma = 10  # integral blows up when this is smaller than 0.4 and cw=0...
    cw = 1
    we = [(u,v,0.) for (u,v) in edges]

    global m = LLROptModel(;
        weighted_edges = we,
        input_edges = input_edges,
        on_ind = on_ind,
        c = cr,
        dc = dc,
        n = n,
        k = k,
        idist = idist,
        ind_set = off_ind,
        idist_set = [1,0,0]
    )

    global d_c = ForwardAlgModel(;
        m = 6,
        set_start = 1,
        comp_arrive = [3,4,5],
        eps = 1e-30
    )

    global d_cdc = ForwardAlgModel(;
        m = 6,
        set_start = 1,
        comp_arrive = [3,4,5],
        eps = 1e-30
    )

    #                        kboff, konb, kuon, kbon, koffu, kuoff, konu
    global logrates = log10.([1,    1,    1,    1e-4,  1,    0,     0])

    args = [d_c, d_cdc, m, cr, gamma, cw]

    # target_mle_exact(logrates, args...);
    initForwardRates!(logrates, args)
    # print("\n")
    # show(tauOff(logrates, cr, cw, gamma))

    # t_off = 1
    # t_on = 0.0001

    # Pjoint!(t_off, t_on, d_c)

    # plotting 
    # global tvals = range(0,20/d_c.maxeigval_set,length=1000)
    # global density = [outer_integrand(t, d_c, d_cdc, m) for t in tvals]

    ## toff = 1
    # @assert isapprox(d_c.F_comp[1,3], 0.11652126742756937, atol=1e-8) 
    # @assert isapprox(d_c.F_comp[1,4], 0.13483690631630374, atol=1e-8) 
    # @assert isapprox(d_c.F_comp[1,5], 0.11652126742756937, atol=1e-8) 

    ## t_on = 1
    # @assert isapprox(d_c.F_set[3,1], 0.21390913026027933, atol=1e-8) 
    # @assert isapprox(d_c.F_set[4,1], 0.18053603312097147, atol=1e-8) 
    # @assert isapprox(d_c.F_set[5,1], 0.21390913026027933, atol=1e-8) 

    ## pure density function
    # @assert isapprox(outer_integrand_test(1, d_c), 0.3678794411714421, atol=1e-8) 
    # @assert isapprox(outer_integrand_test(4, d_c), 0.018315638888734224, atol=1e-8) 
    # @assert isapprox(outer_integrand_test(5, d_c), 0.006737946999085455, atol=1e-8)

    global tauOffVal = tauOff(logrates, cr, cw, gamma)
    global tauOnVal = tauOn(logrates, cr, cw, gamma)

    # integral, _ = quadde(t -> outer_integrand_test(t, d_c), 0, 20/d_c.maxeigval_set, rtol=1e-10)

    xmin = [0,0]
    xmax = [30/d_c.maxeigval_comp, 30/d_c.maxeigval_set]
    integral, err = hcubature(t -> t[2]*PjointVec(t, d_c), xmin, xmax; reltol=1e-10, maxevals=8000000)

    @show integral, err
    @show tauOnVal
end

# things to try:
#     - cubature package
#     - validate the marginals: test each integral separately with mathematica


# reproducing the error


function f(t,a,b,c,d)
    return (((a^2/(a+b))*exp(-a*t[1]) + (b^2/(a+b))*exp(-b*t[1]))
           *((c^2/(c+d))*exp(-c*t[2]) + (d^2/(c+d))*exp(-d*t[2])))
end

let
    # max(20/a, 20/b),max(20/c, 20/d)

    a = 0.1
    b = 10
    c = 4
    d = 1

    int, err = hcubature(t -> t[2]*f(t, a, b, c, d), 
                        [0,0], [max(20/a, 20/b),max(20/c, 20/d)]; reltol=1e-15, maxevals=20000)

end


@time let 
    using QuadGK

    # Our integrand:
    f(x, y) = exp(-(x^2 + y^2))

    # A helper that integrates f over y from c to d, keeping x fixed:
    function inner_integral(x, c, d)
        integral_value, _ = quadgk(y -> f(x, y), c, d)
        return integral_value
    end

    # Main function to perform the double integral:
    function double_integral(a, b, c, d)

        # We define a function of x that integrates over y:
        integrand(x) = inner_integral(x, c, d)

        # Then integrate over x:
        outer_value, _ = quadgk(integrand, a, b)
        return outer_value
    end

    # Example usage: integrate exp(-(x^2 + y^2)) for x ∈ [-2,2] and y ∈ [-1,1]
    a, b = -Inf, Inf
    c, d = -Inf, Inf
    result = double_integral(a, b, c, d)
    print("Result of the double integral = ", result)

end