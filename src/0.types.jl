# constant memory objects
export LLROptModel
Base.@kwdef mutable struct LLROptModel
    # input params
    weighted_edges
    input_edges
    c
    dc
    on_ind
    idist
    n
    # k::Int = 1
    
    # build_mat
    Q_c = nothing; # to initialize
    Q_cdc = nothing; # to initialize

    # generate
    t = nothing; # to initialize
    s = nothing; # to initialize

    # get_residence_time
    s_in_on = nothing; # to initialize
    rt = nothing; # to initialize
    onrt = nothing; # to initialize
    offrt = nothing; # to initialize

    # indicators
    off_ind = nothing; # to initialize
    _idist0_on = nothing; # to initialize
    _idist0_off = nothing; # to initialize

    # ll_full
    ind_set = nothing; # to initialize
    t_set = nothing; # to initialize
    t_comp = nothing; # to initialize
    idist_set = nothing; # to initialize

    ll_c = nothing; # to initialize
    ll_cdc = nothing; # to initialize
    llr = nothing; # to initialize
    ir = nothing; # to initialize

end

import Base.show
function Base.show(io::IO, m::LLROptModel)
    print(io, "LLROptModel(...)")
end

