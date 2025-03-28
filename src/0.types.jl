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
    k
    
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
    jc = nothing; # to initialize
    bc = nothing; # to initialize

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

export ForwardAlgModel
Base.@kwdef mutable struct ForwardAlgModel
    m
    set_start
    comp_arrive
    eps

    # indicator array for set
    ind_set = nothing;
    ind_comp = nothing;
    _set = nothing;
    _comp = nothing;

    # rate matrix
    Q = nothing;

    # set
    D_set = nothing;
    W_set = nothing;
    W_set_inv = nothing;
    aux_set = nothing;
    P_set = nothing;
    Q_h_set = nothing;
    F_set = nothing;

    # comp
    D_comp = nothing;
    W_comp = nothing;
    W_comp_inv = nothing;
    aux_comp = nothing;
    P_comp = nothing;
    Q_h_comp = nothing;
    F_comp = nothing;

    # likelihood
    Pjoint = nothing;

    # log-likelihoods
    logPc = nothing;
    logPcdc = nothing;

end

import Base.show
function Base.show(io::IO, m::LLROptModel)
    print(io, "LLROptModel(...)")
end

