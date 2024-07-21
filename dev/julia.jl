## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
using MethodAnalysis

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
function fun3(a)
    return a
end

# . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
function fun4()
    c = 0
    dat = Vector{Any}(zeros(Int(1e6)))
    for it in dat
        # who is it
        c += fun3(it)
    end
    c
end

function fun5()
    c = 0
    dat = Vector{Float64}(zeros(Int(1e6)))
    for it in dat
        c += fun3(it)
    end
    c
end

## . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
@time fun4()
@time fun5()