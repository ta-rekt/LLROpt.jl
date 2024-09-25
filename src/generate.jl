# . - -- .- .-.- . .- .-- -.- .- .- --.-.... - -.-
# TODO: Finish this shit
export generate!
function generate!(m::LLROptModel)

    m.s, m.t = zeros(Int, m.n), zeros(m.n)
    
    m.s[1] = sample(eachindex(m.idist), Weights(m.idist))
    q = -m.Q_c[m.s[1], m.s[1]]

    _DD = _ddist(m.Q_c)

    for i in 1:m.n-1
        si = m.s[i]
        q = -m.Q_c[si, si]

        # t
        e = Exponential(1/q)
        m.t[i] = rand(e)

        # s        
        _dd = _DD[si, :]
        m.s[i+1] = sample(eachindex(_dd), Weights(_dd))
    end

    # t
    si = m.s[end]
    q = -m.Q_c[si, si]
    e = Exponential(1/q)
    m.t[end] = rand(e)
end

# function generate!(m::LLROptModel)
#     # TODO: inline
#     s, t = generate(m.n, m.Q_c, m.idist)
    
#     m.s = s
#     m.t = t
#     return m
# end