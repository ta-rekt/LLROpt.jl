using LLROpt
using Test

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

    # build_P
    let
        weighted_edges = [(1, 2, 0.5), (2, 1, 2.0)]
        input_edges = [(1, 2)]
        c = 4.0
        
        # params:
        Q = build_mat(weighted_edges, input_edges, c)
        t = 1e20
    
        _P = P(Q, t)
        @test _P[1] == 0.5
        @test _P[2] == 0.5
        @test _P[3] == 0.5
        @test _P[4] == 0.5
    end

    # fpt
    let
        weighted_edges = [(1, 2, 0.5), (2, 1, 2.0)]
        input_edges = [(1, 2)]
        c = 4.0
        
        # params:
        Q = build_mat(weighted_edges, input_edges, c)
        
        for t in range(0.0, 100.0; length = 100)
            hset = [rand(1:2)]
            QhPQht = fpt(Q, hset, t)
            QpQt = Q * P(Q, t)
            @test sum(QpQt .- QhPQht) < 1e-6
        end
    end

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
    let
        weighted_edges = [(1, 2, 2.0), (2, 1, 2.0)]
        input_edges = [(1, 2)]
        c = 1.0
        
        # params:
        on_ind = [0, 1]
        _sharpness = sharpness(weighted_edges, input_edges, c, on_ind)
        @test _sharpness == 0.25
    end

    # ir_acc
    let
        weighted_edges = [(1, 2, 2.0), (2, 1, 2.0)]
        input_edges = [(1, 2)]
        c = 1.0
        
        # params:
        on_ind = [0, 1]
        dc = c * 1e-1
        _ir_acc = ir_acc(weighted_edges, input_edges, c, dc, on_ind)
        @test _ir_acc ≈ 0.002326968666254386
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
        w = ll_full(weighted_edges, input_edges, c, ind_set, t_set, t_comp, idist_set)
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
        n = 1000000
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


end
