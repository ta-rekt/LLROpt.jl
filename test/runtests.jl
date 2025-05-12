using LLROpt
using Test
using Cubature
using JSON

@testset "LLROpt.jl" begin

    let 
        ## circuit specifications file
        projectdir = "/home/tohme/.julia/dev/LLROpt/"
        circuits_file = string(projectdir,"spec/circuits.json")
        global circuit_specs = Dict()

        open(circuits_file, "r") do f
            circuit_specs = JSON.parse(f)
        end

        ## ----.-.-.--....-.---.-..-.-.-.-.-.--.-.-.-----.--
        ## testing 2bs circuit

        # loading testset
        tests_file = string(projectdir, 
                    "test/data/20250512-2bs-fir-free-c0-testset.json")
        testset = Dict()

        open(tests_file, "r") do f
            testset = JSON.parse(f)
        end

        for kc in ["1", "5", "10", "15", "20", "25", "30"]

            kg = "1"
            kw = "1"

            # load circuit specifications
            circuitId = testset["circuit_id"]
            global current_test = testset[kc][kg][kw]
            logrates = current_test["logrates"]
            cr = current_test["c0"]
            t1 = current_test["t1"]
            t2 = current_test["t2"]

            circuit_2bs = circuit_specs[circuitId]
            edges = circuit_2bs["edges"]
            input_edges = circuit_2bs["input_edges"]
            nstates = circuit_2bs["nstates"]
            off_ind = circuit_2bs["off_indicator"]
            off_ind = BitArray(off_ind)
            start_state = circuit_2bs["start_state"]
            hitting_set = circuit_2bs["hitting_set"]

            eps = 1e-30
            reltol = 1e-10
            evalrtol = 1e-6 
            maxevals = 100000

            global d = OneBurstDecoyModel(;
                circuitId = circuitId,
                nstates = nstates,
                edges = edges,
                input_edges = input_edges,
                logrates = logrates,
                cr = cr,
                cw = nothing,
                gamma = nothing,
                ind_set = off_ind,
                start_state = start_state,
                hitting_set = hitting_set,
                eps = eps,
                reltol = reltol,
                maxevals = maxevals,
            )

            d1 = deepcopy(d)
            d2 = deepcopy(d)

            h = 0.001 * d.cr
            d1.cr = d.cr - h
            d2.cr = d.cr - 2*h

            initFull!(d)
            initFull!(d1)
            initFull!(d2)

            # Pon!(t2, d)
            # Poff!(t1, d)
            
            Pjoint!(t1, t2, d)

            Qtest = reshape(vcat(current_test["rate_matrix"]...), (nstates, nstates))'
            Ptest = current_test["p_joint"]
            PssTest = current_test["ss_prob"]
            tOffTest = current_test["tau_off"]
            tOnTest = current_test["tau_on"]
            varTest = current_test["var_rate"]
            sharpTest = current_test["sharpness"]
            firFullTest = current_test["fisher_rate"]
            firAccTest = current_test["fisher_rate_acc"]
            integrandTest = current_test["integrand"]

            @test d.edges == edges
            @test d.input_edges == input_edges
            @test d.nstates == nstates
            @test d.ind_set == off_ind
            @test d.start_state == start_state
            @test d.hitting_set == hitting_set

            @test d.logrates == logrates
            @test d.cr == cr

            for (i,j) in zip(range(1, nstates), range(1, nstates))
                try
                    @test isapprox(d.Q[i,j], Qtest[i,j], rtol = evalrtol)
                catch e
                    # println("i: ", i, " j: ", j)
                    # println("Q: ", d.Q[i,j])
                    # println("Qtest: ", Qtest[i,j])
                    # println("evalrtol: ", evalrtol)
                end
            end

            @test isapprox(tauOff(d), tOffTest, rtol = evalrtol)
            @test isapprox(tauOn(d), tOnTest, rtol = evalrtol)
            @test isapprox(P_ss(d), PssTest, rtol = evalrtol)
            @test isapprox(variance(d), varTest, rtol = evalrtol)
            @test isapprox(d.Pjoint, Ptest, rtol = evalrtol)
            @test isapprox(sharpness(d), sharpTest, rtol = 1e-4)
            @test isapprox(FisherRateAcc(d), firAccTest, rtol = 1e-4)
            @test isapprox(integrand([t1, t2], d, d1, d2), integrandTest, rtol = 1e-4)
            @test isapprox(FisherRate(d, d1, d2), firFullTest, rtol = 1e-4)
        end
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

    # # . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
    # let
    #     Q = [-2 1 1; 1 -3 2; 0 0.5 -0.5]
    #     idist = [1, 1, 1] ./ 3
    #     n = 2000000
    #     s, t = generate(n, Q, idist)
    #     k = size(Q, 1)
    #     samplesd_P_ss = zeros(k)
    #     for (si, ti) in zip(s, t)
    #         samplesd_P_ss[si] += ti
    #     end
    #     samplesd_P_ss ./= sum(samplesd_P_ss)

    #     anal_P_ss = P_ss(Q)
        
    #     @test maximum(abs, samplesd_P_ss .- anal_P_ss) < 1e-3
    # end

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
        @test onrt == [4.0, 5.0, 1.0]
        @test offrt == [2.5, 1.0]
        @test sum(t) == sum(onrt) + sum(offrt)
        @test sum(t) == sum(rt)
    end

end
