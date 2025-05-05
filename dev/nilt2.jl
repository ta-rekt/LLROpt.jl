# nilt2.jl  –  two–variable numerical inverse Laplace transform
# -------------------------------------------------------------
using QuadGK          # standard package (`] add QuadGK` once)

# ---------- 1‑D Bromwich integral (Talbot contour, scalar t) -------------
function nilt1(F::Function, t::Real; σ = nothing, N = 32)
    t ≤ 0   && error("t must be positive")
    σ === nothing && (σ = 1 + abs(real(F(1 + 0im))))  # crude safe shift
    f(θ) = begin
        z      = θ / tan(θ) + im*θ           # Talbot mapping
        s      = σ + z / t                   # rescale for given t
        dsdθ   = (im * (1 + θ/tan(θ) - θ^2 / sin(θ)^2)) / t
        real(exp(s * t) * F(s) * dsdθ)       # integrand on (0,π)
    end
    integral, _ = quadgk(f, 0, π; atol = 1e-12, rtol = 1e-12, maxevals = N)
    return integral / π
end

# ---------- 2‑D inverse Laplace by successive 1‑D inversions -------------
"""
    nilt2(F, t1, t2; σ1 = nothing, σ2 = nothing, N = 32)

Compute the two‑variable inverse Laplace transform of `F(s1,s2)` at
positive times `t1`, `t2` by first inverting in `s1` (with `s2` numeric)
and then in `s2`.
"""
function nilt2(F::Function, t1::Real, t2::Real; σ1 = nothing,
               σ2 = nothing, N = 32)

    t1 ≤ 0 && error("t1 must be positive")
    t2 ≤ 0 && error("t2 must be positive")

    # inner inverse in s1, s2 held numeric -------------------------------
    inner(s2num) = nilt1(s1 -> F(s1, s2num), t1; σ = σ1, N = N)

    # outer inverse in s2 ------------------------------------------------
    return nilt1(s2 -> inner(s2), t2; σ = σ2, N = N)
end

# ----------------------- Self‑test --------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    F(s1,s2) = 1 / ((s1 + 2) * (s2 + 3))          # Laplace domain
    for (t1,t2) in ((0.5,1.0),(1.0,1.0),(2.0,3.0))
        num = nilt2(F, t1, t2)
        exact = exp(-2t1)*exp(-3t2)
        println("t1=$(t1)  t2=$(t2)   nILT = $(num)   exact = $(exact)")
    end
end
