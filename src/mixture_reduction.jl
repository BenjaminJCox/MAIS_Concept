using LinearAlgebra
using Distributions
using InvertedIndices

include("distances_normal.jl")

function construct_distance_matrix(proposals, distance)
    n_proposals = length(proposals)
    distance_matrix = Matrix{Float64}(undef, n_proposals, n_proposals)
    idxs = 1:n_proposals
    for col in idxs, row in idxs
        distance_matrix[row, col] = distance(proposals[row], proposals[col])
    end
    return distance_matrix
end

# function _corrupt_diagonal!(distance_matrix)
#     n_proposals = size(distance_matrix, 1)
#     distance_matrix .+= diagm([Inf for i = 1:n_proposals])
#     return distance_matrix
# end

function merge_normals(N1, N2; λ = 0.5)
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

function merge_hellinger!(proposals, λ = 0.1)
    dm = construct_distance_matrix(proposals, pw_hellinger) .+ diagm([Inf for i = 1:length(proposals)])
    proposals_scratch = copy(proposals)
    _temp_proposals = Vector{MvNormal}()
    while minimum(dm) <= λ
        _temp_proposals = Vector{MvNormal}()
        to_merge = argmin(dm)
        N1 = proposals_scratch[to_merge[1]]
        N2 = proposals_scratch[to_merge[2]]
        merge_dist = merge_normals(N1, N2)
        proposals_scratch = proposals_scratch[Not(to_merge[1], to_merge[2])]
        push!(_temp_proposals, merge_dist)
        append!(_temp_proposals, proposals_scratch)
        proposals_scratch = _temp_proposals
        dm = construct_distance_matrix(_temp_proposals, pw_hellinger) .+= diagm([Inf for i = 1:length(_temp_proposals)])
    end
    return proposals_scratch
end

function mixture_culling_ess!(proposals, samples_each, weights; ν = 0.7)
    n_proposals = length(proposals)
    g_ess = inv(sum(weights .^ 2))
    _temp_proposals = Vector{MvNormal}()
    l_ess = Vector{Float64}(undef, n_proposals)
    for p_idx in 1:n_proposals
        s_offset = (p_idx - 1) * samples_each
        p_weights = weights[(s_offset+1):(s_offset+samples_each)]
        pwess = p_weights ./ sum(p_weights)
        l_ess[p_idx] = inv(sum(pwess .^ 2))
        if l_ess[p_idx] > ν * samples_each
            push!(_temp_proposals, proposals[p_idx])
        end
    end
    proposals = _temp_proposals
end
