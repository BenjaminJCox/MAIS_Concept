using Distributions
using LinearAlgebra
using RecursiveArrayTools
# using LatinHypercubeSampling

include("weighting_schemes.jl")

# function init_proposals(dim, n_proposals; lims = (-2, 2), σ = 1.0)
#     proposals = Vector{MvNormal}(undef, n_proposals)
#     for p_idx = 1:n_proposals
#         proposals[p_idx] = MvNormal(rand(Uniform(lims...), dim), Matrix(σ .* I(dim)))
#     end
#     return proposals
# end


function dmpmc_step!(proposals::Vector{MvNormal}, target::Function, samples_each::Int; global_adapt = true)
    # we assume the proposals are normal, although we could modify this to not be the case: usual multivariate t may be better
    n_proposals = length(proposals)
    samples = Vector{Vector{Float64}}(undef, n_proposals * samples_each)
    wts = Vector{Float64}(undef, n_proposals * samples_each)
    # Threads.@threads for p_idx = 1:n_proposals
    for p_idx = 1:n_proposals
        prop = proposals[p_idx]
        s_offset = (p_idx - 1) * samples_each
        for i = 1:samples_each
            samples[s_offset+i] = rand(prop)
            # wts[s_offset+i] = dm_weights(samples[s_offset+i], proposals, target)
        end
        wts[(s_offset+1):(s_offset+samples_each)] = dm_weights_new(samples[(s_offset+1):(s_offset+samples_each)], proposals, target)
    end
    weights = Weights(wts)
    if global_adapt
        locations = sample(samples, weights, n_proposals; replace = true)
        Threads.@threads for p_idx = 1:n_proposals
            Σ = proposals[p_idx].Σ
            proposals[p_idx] = MvNormal(locations[p_idx], Σ)
        end
    else
        Threads.@threads for p_idx = 1:n_proposals
            s_offset = (p_idx - 1) * samples_each
            p_samples = samples[(s_offset+1):(s_offset+samples_each)]
            p_weights = weights[(s_offset+1):(s_offset+samples_each)]
            location = sample(p_samples, p_weights, 1; replace = true)[1]
            Σ = proposals[p_idx].Σ
            proposals[p_idx] = MvNormal(location, Σ)
        end
    end
    return @dict samples weights
end

function dmpmc_init_adapt!(proposals, target; n_adapts = 20, samples_adapts = 50, global_adapt = false)
    for i = 1:n_adapts
        dmpmc_step!(proposals, target, samples_adapts; global_adapt = global_adapt)
    end
    return proposals
end
