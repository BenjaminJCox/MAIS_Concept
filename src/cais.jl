using LinearAlgebra
using Distributions

include("weighting_schemes.jl")

function cais_step!(proposals::Vector{MvNormal}, target::Function, samples_each::Int; N_t = 0.1 * samples_each, γ = 3)
    # we assume the proposals are normal, although we could modify this to not be the case: usual multivariate t may be better
    n_proposals = length(proposals)
    samples = Vector{Vector{Float64}}(undef, n_proposals * samples_each)
    wts = Vector{Float64}(undef, n_proposals * samples_each)
    Threads.@threads for p_idx = 1:n_proposals
        prop = proposals[p_idx]
        s_offset = (p_idx - 1) * samples_each
        # samples[(s_offset + 1):(s_offset + samples_each)] = [rand(prop) for i in 1:samples_each]
        for i = 1:samples_each
            samples[s_offset+i] = rand(prop)
            # wts[s_offset+i] = dm_weights(samples[s_offset+i], proposals, target)
        end
        wts[(s_offset+1):(s_offset+samples_each)] = dm_weights_new(samples[(s_offset+1):(s_offset+samples_each)], proposals, target)
    end
    weights = Weights(wts)
    Threads.@threads for p_idx = 1:n_proposals
        s_offset = (p_idx - 1) * samples_each
        p_samples = samples[(s_offset+1):(s_offset+samples_each)]
        p_weights = weights[(s_offset+1):(s_offset+samples_each)]


        _ess = inv(sum(p_weights .^ 2))

        if _ess <= N_t
            n_weights = p_weights .^ inv(γ)
            n_weights ./= sum(n_weights)
        else
            n_weights = p_weights ./ sum(p_weights)
        end

        is_mean = sum(n_weights .* p_samples)

        is_mdiff = p_samples .- [is_mean for i = 1:samples_each]

        is_cov = sum(n_weights .* [i * i' for i in is_mdiff])

        W = 1.0 - sum(n_weights .^ 2)
        is_cov = inv(W) .* sum(n_weights .* [i * i' for i in is_mdiff])

        is_cov = is_cov .+ is_cov'
        is_cov ./= 2

        Σ = is_cov
        μ = is_mean

        proposals[p_idx] = MvNormal(μ, Σ)
    end
    return @dict samples weights
end
