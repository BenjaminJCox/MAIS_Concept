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
    # μ1, μ2 = N1.μ, N2.μ
    @assert length(N1.μ) == length(N2.μ)
    k = length(N2.μ)
    # Σ1, Σ2 = N1.Σ, N2.Σ
    μres = 0.5 .* (N1.μ .+ N2.μ)
    # Σres = 0.25 .* (Σ1 .+ Σ2)
    # Σres = Σ1 .+ Σ2
    Σres = λ .* (N1.Σ .+ N2.Σ)
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
    proposals = proposals_scratch
    # return proposals_scratch
end

function mixture_culling_ess!(proposals, samples_each, weights; ν = 0.8)
    n_proposals = length(proposals)
    # g_ess = inv(sum(weights .^ 2))
    _temp_proposals = Vector{MvNormal}()
    for p_idx in 1:n_proposals
        s_offset = (p_idx - 1) * samples_each
        @views p_weights = weights[(s_offset+1):(s_offset+samples_each)]
        pw_ess = inv(sum(x -> x .^2, p_weights ./ sum(p_weights)))
        if pw_ess > ν * samples_each
            push!(_temp_proposals, proposals[p_idx])
        end
    end
    proposals = _temp_proposals
end

function split_ess!(proposals, samples_each, weights; ν = 0.2)
    n_proposals = length(proposals)
    gwess = weights ./ sum(weights)
    # g_ess = inv(sum(gwess .^ 2))

    l_ess = zeros(n_proposals, n_proposals)
    for p_idx in 1:n_proposals
        s_offset = (p_idx - 1) * samples_each
        @views p_weights = weights[(s_offset+1):(s_offset+samples_each)]
        pw_ess = inv(sum(x -> x .^2, p_weights ./ sum(p_weights)))
        l_ess[:, p_idx] .+= pw_ess
        l_ess[p_idx, :] .+= pw_ess
    end
    _temp_proposals = copy(proposals)
    for p1_idx in 1:n_proposals
        for p2_idx in p1_idx:n_proposals
            if (p1_idx != p2_idx) && (l_ess[p1_idx, p2_idx] < 2 * ν * samples_each)
                push!(_temp_proposals, merge_normals(proposals[p1_idx], proposals[p2_idx]))
            end
        end
    end
    proposals = _temp_proposals
end
