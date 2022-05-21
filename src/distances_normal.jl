using LinearAlgebra
using Distributions

function pw_kldiv(N1, N2)
    μ1, μ2 = N1.μ, N2.μ
    @assert length(μ1) == length(μ2)
    k = length(μ1)
    Σ1, Σ2 = N1.Σ, N2.Σ
    kld = 0.5 .* (tr(inv(Σ2) * Σ1) + (μ2 - μ1)' * inv(Σ2) * (μ2 - μ1) - k + log(det(Σ2) / det(Σ1)))
    return kld
end

function pw_wasserstein(N1, N2)
    μ1, μ2 = N1.μ, N2.μ
    @assert length(μ1) == length(μ2)
    k = length(μ1)
    Σ1, Σ2 = N1.Σ, N2.Σ
    Σ2_rt = sqrt(Matrix(Σ2))
    wass = norm(μ1 .- μ2) .^ 2.0 + tr(Σ1 .+ Σ2 .- 2.0 .* sqrt(Σ2_rt * Σ1 * Σ2_rt))
    return wass
end

function pw_bhattacharyya(N1, N2)
    μ1, μ2 = N1.μ, N2.μ
    @assert length(μ1) == length(μ2)
    k = length(μ1)
    Σ1, Σ2 = N1.Σ, N2.Σ
    bhat =
        0.125 .* (μ1 .- μ2)' * inv(0.5 .* (Σ1 .+ Σ2)) * (μ1 .- μ2) .+
        0.5 .* (log(det(0.5 .* (Σ1 .+ Σ2))) .- 0.5 .* (log(det(Σ1)) + log(det(Σ2))))
    return bhat[1]
end

function pw_bhatt_coeff(N1, N2)
    return exp(-pw_bhattacharyya(N1, N2))
end

function pw_sqhellinger(N1, N2)
    μ1, μ2 = N1.μ, N2.μ
    @assert length(μ1) == length(μ2)
    k = length(μ1)
    Σ1, Σ2 = N1.Σ, N2.Σ
    t1 = ((det(Σ1) .* det(Σ2)) .^ (0.25)) ./ sqrt(det(0.5 .* (Σ1 .+ Σ2)))
    t2 = (μ1 .- μ2)' * inv(0.5 .* (Σ1 .+ Σ2)) * (μ1 .- μ2)
    return 1.0 .- t1 .* exp(-0.125 .* t2)
end

function pw_hellinger(N1, N2)
    return sqrt(abs(pw_sqhellinger(N1, N2)))
end
