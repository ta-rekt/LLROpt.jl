using LLROpt
using Test
using Cubature

@testset "LLROpt.jl" begin

    # build_mat
    let
        weighted_edges = [(1, 2, 0.5), (2, 1, 2.0)]
        input_edges = [(1, 2)]
        c = 3.0
        Q = build_mat(weighted_edges, input_edges, c)
        
        N = size(Q, 1)
        for ri in 1:N
            @test sum(view(Q, ri, :)) == 0.0
        end
    
        @test Q[1,1] == -1.5
        @test Q[1,2] == 1.5
        @test Q[2,1] == 2.0
        @test Q[2,2] == -2.0
        Q
    end

    # # build_P
    # let
    #     weighted_edges = [(1, 2, 0.5), (2, 1, 2.0)]
    #     input_edges = [(1, 2)]
    #     c = 4.0
        
    #     # params:
    #     Q = build_mat(weighted_edges, input_edges, c)
    #     t = 1e20
    
    #     _P = P(Q, t)
    #     @test _P[1] == 0.5
    #     @test _P[2] == 0.5
    #     @test _P[3] == 0.5
    #     @test _P[4] == 0.5
    # end

    # # fpt
    # let
    #     weighted_edges = [(1, 2, 0.5), (2, 1, 2.0)]
    #     input_edges = [(1, 2)]
    #     c = 4.0
        
    #     # params:
    #     Q = build_mat(weighted_edges, input_edges, c)
        
    #     for t in range(0.0, 100.0; length = 100)
    #         hset = [rand(1:2)]
    #         QhPQht = fpt(Q, hset, t)
    #         QpQt = Q * P(Q, t)
    #         @test sum(QpQt .- QhPQht) < 1e-6
    #     end
    # end

    # P_ss
    let
        weighted_edges = [(1, 2, 0.5), (2, 1, 2.0)]
        input_edges = [(1, 2)]
        c = 4.0
        
        # params:
        Q = build_mat(weighted_edges, input_edges, c)
        _P_ss = P_ss(Q)
        @test _P_ss[1] == 0.5
        @test _P_ss[2] == 0.5
    end

    # EP_ss
    let
        weighted_edges = [(1, 2, 0.5), (2, 1, 2.0)]
        input_edges = [(1, 2)]
        c = 4.0
        
        # params:
        Q = build_mat(weighted_edges, input_edges, c)
        @test EP_ss(Q) == 0.0
    end

    # Z
    let
        weighted_edges = [(1, 2, 0.5), (2, 1, 2.0)]
        input_edges = [(1, 2)]
        c = 4.0
        
        # params:
        Q = build_mat(weighted_edges, input_edges, c)
        _Z = Z(Q)
        @test _Z[1] ≈ 0.125
        @test _Z[2] ≈ -0.125
        @test _Z[3] ≈ -0.125
        @test _Z[4] ≈ 0.125
    end

    ## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
    # pi_on
    let
        weighted_edges = [(1, 2, 2.0), (2, 1, 2.0)]
        input_edges = [(1, 2)]
        c = 1.0
        
        # params:
        on_ind = [0, 1]
        _pi_on = pi_on(weighted_edges, input_edges, c, on_ind)
        @test _pi_on == 0.5
    end

    ## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
    # variance
    let
        weighted_edges = [(1, 2, 2.0), (2, 1, 2.0)]
        input_edges = [(1, 2)]
        c = 1.0
        
        # params:
        on_ind = [0, 1]
        _var = variance(weighted_edges, input_edges, c, on_ind)
        @test _var == 0.125
    end

    ## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
    # sharpness
    # let
    #     weighted_edges = [(1, 2, 2.0), (2, 1, 2.0)]
    #     input_edges = [(1, 2)]
    #     c = 1.0
        
    #     # params:
    #     on_ind = [0, 1]
    #     _sharpness = sharpness(weighted_edges, input_edges, c, on_ind)
    #     @test _sharpness == 0.25
    # end

    # ir_acc
    let
        weighted_edges = [(1, 2, 2.0), (2, 1, 2.0)]
        input_edges = [(1, 2)]
        c = 1.0
        
        # params:
        on_ind = [0, 1]
        dc = c * 1e-1
        _ir_acc = ir_acc(weighted_edges, input_edges, c, dc, on_ind)
        @test isapprox(_ir_acc, 0.002326968666254386, rtol=1e-6)
    end

    # . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
    # ll_full
    let
        weighted_edges = [(1, 2, 2.0), (2, 1, 2.0)]
        input_edges = [(1, 2)]
        c = 1.0
        
        # params:
        ind_set = [0, 1]
        t_set = [1, 2, 3]
        t_comp = [2, 3, 4]
        idist_set = [0.5]
        Q = build_mat(weighted_edges, input_edges, c)
        w = ll_full(Q, ind_set, t_set, t_comp, idist_set)
        # Tested against python version
        @test w ≈ [-0.6931471805599453, -1.3068528194400546, -3.3068528194400524, -3.3068528194400524, -5.3068528194400475, -5.3068528194400475, -7.306852819440041]
    end

    # . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
    let
        Q = [-2 1 1; 1 -3 2; 0 0.5 -0.5]
        _DD = LLROpt._ddist(Q)
        @test _DD ≈ [
            0.0          0.5  0.5;
            0.333333333  0.0  0.666666666;
            0.0          1.0  0.0;
        ]
    end

    # . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
    let
        Q = [-2 1 1; 1 -3 2; 0 0.5 -0.5]
        idist = [1, 1, 1] ./ 3
        n = 2000000
        s, t = generate(n, Q, idist)
        k = size(Q, 1)
        samplesd_P_ss = zeros(k)
        for (si, ti) in zip(s, t)
            samplesd_P_ss[si] += ti
        end
        samplesd_P_ss ./= sum(samplesd_P_ss)

        anal_P_ss = P_ss(Q)
        
        @test maximum(abs, samplesd_P_ss .- anal_P_ss) < 1e-3
    end

    ## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
    # _jump_count
    let
        on_ind = BitArray([1,1,0])
        s = [1,2,2,3,1,1,1,2,3,1]
        @test LLROpt._jump_count(s, on_ind) == 5
        s = [1,2,2,3,1,2,1,2,3]
        @test LLROpt._jump_count(s, on_ind) == 4
    end

    ## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
    # get_residence_time
    let
        on_ind = BitArray([1,1,0])
        s = [1,2,2,3,  1,1,1,2,3,1]
        t = [1,2,1,2.5,1,2,1,1,1,1]
        s_in_on, rt, onrt, offrt = get_residence_time(s, t, on_ind)
        @assert onrt == [4.0, 5.0, 1.0]
        @assert offrt == [2.5, 1.0]
        @assert sum(t) == sum(onrt) + sum(offrt)
        @assert sum(t) == sum(rt)
    end

    ## ----.-.-.--....-.---.-..-.-.-.-.-.--.-.-.-----.--
    # On, Off first passage times and means
    let 
        #                 kboff,   konb,  kuon,  kbon, koffu, kuoff, konu
        logrates = log10.([0.0502,  0.2,  0.05,  1e-5, 5,     0,     0])

        edges = [(1, 2), (2, 1), (1, 6), (6, 1), (4, 3), (3, 4), 
                (4, 5), (5, 4), (6, 5), (4, 1), (2, 3), (1, 4)]
        input_edges = [(1,2), (4,3)]
        off_ind = BitArray([1,1,0,0,0,1])
        on_ind = BitArray([0,0,1,1,1,0])
        idist = ones(6) ./ 6
        n = 1
        k = 1
        cr = 50
        dc = cr / 100
        beta = cr
        gamma = 1  # integral blows up when this is smaller than 0.4 and cw=0...
        cw = 1e-5
        eps = 1e-30
        reltol = 1e-12
        maxevals = 100000

        d = OneBurstDecoyModel(;
            nstates = 6,
            edges = edges,
            input_edges = input_edges,
            logrates = logrates,
            cr = cr,
            cw = cw,
            gamma = gamma,
            ind_set = off_ind,
            start_state = 1,
            hitting_set = [3,4,5],
            eps = eps,
            reltol = reltol,
            maxevals = maxevals,
        )

        ddc = OneBurstDecoyModel(;
            nstates = 6,
            edges = edges,
            input_edges = input_edges,
            logrates = logrates,
            cr = cr + dc,
            cw = cw,
            gamma = gamma,
            ind_set = off_ind,
            start_state = 1,
            hitting_set = [3,4,5],
            eps = eps,
            reltol = reltol,
            maxevals = maxevals,
        )

        initDecoys!(d)

        @test isapprox(tauOn(d), 20.2020000004, rtol=1e-8)
        @test isapprox(tauOff(d), 5.398406294820733, rtol=1e-8)

        # testing Pon, Poff, and the partial Fishers
        integralOn, err = hquadrature(t -> t*Pon(t, d), 
            0, 30/d.maxeigval_set; reltol=d.reltol, maxevals=d.maxevals)
        integralOff, err = hquadrature(t -> t*Poff(t, d), 
            0, 30/d.maxeigval_comp; reltol=d.reltol, maxevals=d.maxevals)

        @test isapprox(integralOn, tauOn(d), rtol=1e-6)
        @test isapprox(integralOff, tauOff(d), rtol=1e-6)

        # ---.-.-.-.-.-.-.-.------.---.-...-.-.-.-.-.-.-.-..-----.-..-.-.-.....
    
        #                      kboff,   konb,    kuon,   kbon, koffu, kuoff, konu
        d.logrates = log10.([0.1,     5,        4,     1e-2,  1e-1,     0,   0])
        d.cr = 1e-2
        d.cw = 10
        d.gamma = 1e-3

        initDecoys!(d)

        # testing Pon, Poff, and the partial Fishers
        integralOn, err = hquadrature(t -> t*Pon(t, d), 
            0, 100/d.maxeigval_set; reltol=d.reltol, maxevals=d.maxevals)
        integralOff, err = hquadrature(t -> t*Poff(t, d), 
            0, 100/d.maxeigval_comp; reltol=d.reltol, maxevals=d.maxevals)

        @test isapprox(tauOn(d), 509.7507495004995, rtol=1e-8) 
        @test isapprox(tauOff(d), 1.1990009990009989, rtol=1e-8) 
        @test isapprox(integralOn, tauOn(d), rtol=1e-6)
        @test isapprox(integralOff, tauOff(d), rtol=1e-6)

        
        # ---.-.-.-.-.-.-.-.------.---.-...-.-.-.-.-.-.-.-..-----.-..-.-.-.....
    
        #                      kboff,   konb,    kuon,   kbon, koffu, kuoff, konu
        d.logrates = log10.([0.0005,   1e-5,       1e-5,     1e-5,  1e-5,     0,   0])
        d.cr = 5
        d.cw = 1e-4
        d.gamma = 1

        initDecoys!(d)

        # testing Pon, Poff, and the partial Fishers
        integralOn, err = hquadrature(t -> t*Pon(t, d), 
            0, 1000/d.maxeigval_set; reltol=d.reltol, maxevals=d.maxevals)
        integralOff, err = hquadrature(t -> t*Poff(t, d), 
            0, 100/d.maxeigval_comp; reltol=d.reltol, maxevals=d.maxevals)

        @test isapprox(integralOn, tauOn(d), rtol=1e-6)
        @test isapprox(integralOff, tauOff(d), rtol=1e-6)

    end


    # Fisher information rate
    let 
        nstates = 6
        edges = [(1, 2), (2, 1), (1, 6), (6, 1), (4, 3), (3, 4), 
                 (4, 5), (5, 4), (6, 5), (4, 1), (2, 3), (1, 4)]
        input_edges = [(1,2), (4,3)]
        off_ind = BitArray([1,1,0,0,0,1])
        n = 1
        k = 1
        cr = 50
        dc = cr / 100
        beta = cr
        gamma = 1 
        cw = 1e-3
    
        maxevals = 10000
        reltol = 1e-10
        eps = 1e-30
    
        #                        kboff,   konb,    kuon,   kbon, koffu, kuoff, konu
        global logrates = log10.([0.0502,  0.2,    0.05,   1e-5,  5,     0,     0])
    
        global d_c = OneBurstDecoyModel(;
            nstates = nstates,
            edges = edges,
            input_edges = input_edges,
            logrates = logrates,
            cr = cr,
            cw = cw,
            gamma = gamma,
            ind_set = off_ind,
            start_state = 1,
            hitting_set = [3,4,5],
            eps = eps,
            reltol = reltol,
            maxevals = maxevals,
        )
    
        global d_cdc = OneBurstDecoyModel(;
            nstates = nstates,
            edges = edges,
            input_edges = input_edges,
            logrates = logrates,
            cr = cr + dc,
            cw = cw,
            gamma = gamma,
            ind_set = off_ind,
            start_state = 1,
            hitting_set = [3,4,5],
            eps = eps,
            reltol = reltol,
            maxevals = maxevals,
        )
    
        initDecoys!(d_c)
        initDecoys!(d_cdc)
    
        @test isapprox(FisherRate(d_c, d_cdc), 0.0012884521978649768, rtol = 0.05)
    
        # ---.-.-.-.-.-.-.-.------.---.-...-.-.-.-.-.-.-.-..-----.-..-.-.-.....
    
        #                      kboff,   konb,    kuon,   kbon, koffu, kuoff, konu
        d_c.logrates = log10.([0.1,     5,        4,     1e-2,  1e-1,     0,   0])
        d_c.cr = 5
        d_cdc.logrates = d_c.logrates
        d_cdc.cr = d_c.cr * 1.01
    
        initDecoys!(d_c)
        initDecoys!(d_cdc)
    
        @test isapprox(FisherRate(d_c, d_cdc), 0.07876075503682617, rtol = 0.05)
    
        # ---.-.-.-.-.-.-.-.------.---.-...-.-.-.-.-.-.-.-..-----.-..-.-.-.....
    
        #                      kboff,   konb,    kuon,   kbon, koffu, kuoff, konu
        d_c.logrates = log10.([0.0005,   5,       5,     1e-5,  5,     0,   0])
        d_c.cr = 5000
        d_cdc.logrates = d_c.logrates
        d_cdc.cr = d_c.cr * 1.01
    
        initDecoys!(d_c)
        initDecoys!(d_cdc)
    
        @test isapprox(FisherRate(d_c, d_cdc), 0.806693810766622, rtol = 0.05)
    
        # ---.-.-.-.-.-.-.-.------.---.-...-.-.-.-.-.-.-.-..-----.-..-.-.-.....
    
        #                      kboff,   konb,    kuon,   kbon, koffu, kuoff, konu
        d_c.logrates = log10.([0.0005,   1e-5,       1e-5,     1e-5,  1e-5,     0,   0])
        d_c.cr = 5000
        d_cdc.logrates = d_c.logrates
        d_cdc.cr = d_c.cr * 1.01
    
        initDecoys!(d_c)
        initDecoys!(d_cdc)
    
        @test isapprox(FisherRate(d_c, d_cdc), 1.9973865408615845e-09, rtol = 0.05)

    end
end
