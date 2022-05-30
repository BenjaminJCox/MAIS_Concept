using LinearAlgebra
using Distributions
using InvertedIndices

include("distances_normal.jl")
include("mixture_reduction.jl")

function split_normals(N1, N2; λ = 0.5)
    # assuming unweighted
    μ1, μ2 = N1.μ, N2.μ
    @assert length(μ1) == length(μ2)
    k = length(μ1)
    Σ1, Σ2 = N1.Σ, N2.Σ
    μres = 0.5 .* (μ1 .+ μ2)
    # Σres = 0.25 .* (Σ1 .+ Σ2)
    # Σres = Σ1 .+ Σ2
    Σres = λ .* (Σ1 .+ Σ2)
    return MvNormal(μres, Σres)
end

function split_ess(proposals, samples_each, weights; λ = 0.8, ν = 0.2)
    n_proposals = length(proposals)
    gwess = weights ./ sum(weights)
    g_ess = inv(sum(gwess .^ 2))

    l_ess = zeros(n_proposals, n_proposals)
    for p_idx in 1:n_proposals
        s_offset = (p_idx - 1) * samples_each
        p_weights = weights[(s_offset+1):(s_offset+samples_each)]
        pwess = p_weights ./ sum(p_weights)
        pw_ess = inv(sum(pwess .^ 2))
        l_ess[:, p_idx] .+= pw_ess
        l_ess[p_idx, :] .+= pw_ess
    end
    return l_ess
end

function split_hellinger(proposals, λ = 0.8)
    dm = construct_distance_matrix(proposals, pw_hellinger) .+ diagm([Inf for i = 1:length(proposals)])
    proposals_scratch = copy(proposals)
    _temp_proposals = Vector{MvNormal}()
    while minimum(dm) >= λ
        _temp_proposals = Vector{MvNormal}()
        to_split = argmax(dm)
        N1 = proposals_scratch[to_split[1]]
        N2 = proposals_scratch[to_split[2]]
        merge_dist = split_normals(N1, N2)
        push!(_temp_proposals, merge_dist)
        append!(_temp_proposals, proposals_scratch)
        proposals_scratch = _temp_proposals
        dm = construct_distance_matrix(_temp_proposals, pw_hellinger) .+= diagm([Inf for i = 1:length(_temp_proposals)])
    end
    return proposals_scratch
end
