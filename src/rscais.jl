using LinearAlgebra
using Distributions

include("weighting_schemes.jl")

function rscais_step!(proposals::Vector{MvNormal}, target::Function, samples_each::Int; α = 0.1, β = 0.1)
    n_proposals = length(proposals)
    samples = Vector{Vector{Float64}}(undef, n_proposals * samples_each)
    wts = Vector{Float64}(undef, n_proposals * samples_each)
    Threads.@threads for p_idx = 1:n_proposals
        prop = proposals[p_idx]
        s_offset = (p_idx - 1) * samples_each
        for i = 1:samples_each
            samples[s_offset+i] = rand(prop)
            wts[s_offset+i] = dm_weights(samples[s_offset+i], proposals, target)
        end
    end
    weights = Weights(wts)
    Threads.@threads for p_idx = 1:n_proposals
        s_offset = (p_idx - 1) * samples_each
        p_samples = samples[(s_offset+1):(s_offset+samples_each)]
        p_weights = weights[(s_offset+1):(s_offset+samples_each)]

        n_weights = p_weights ./ sum(p_weights)

        is_mean = sum(n_weights .* p_samples)

        is_mdiff = p_samples .- [is_mean for i = 1:samples_each]

        W = 1.0 - sum(n_weights .^ 2)
        is_cov = inv(W) .* sum(n_weights .* [i * i' for i in is_mdiff])

        Σ = (1.0 .- β) .* proposals[p_idx].Σ .+ β .* is_cov
        μ = (1.0 .- α) .* proposals[p_idx].μ .+ α .* is_mean

        proposals[p_idx] = MvNormal(μ, Σ)
    end
    return @dict samples weights
end

function rscais_gradual_step!(
    proposals::Vector{MvNormal},
    target::Function,
    samples_each::Int;
    α = 0.1,
    β = 0.1,
    η = 0.1,
    γ = 3,
    N_t = 0.1 * samples_each,
)
    n_proposals = length(proposals)
    samples = Vector{Vector{Float64}}(undef, n_proposals * samples_each)
    wts = Vector{Float64}(undef, n_proposals * samples_each)
    Threads.@threads for p_idx = 1:n_proposals
        prop = proposals[p_idx]
        s_offset = (p_idx - 1) * samples_each
        for i = 1:samples_each
            samples[s_offset+i] = rand(prop)
            wts[s_offset+i] = dm_weights(samples[s_offset+i], proposals, target)
        end
    end
    weights = Weights(wts)
    Threads.@threads for p_idx = 1:n_proposals
        s_offset = (p_idx - 1) * samples_each
        p_samples = samples[(s_offset+1):(s_offset+samples_each)]
        p_weights = weights[(s_offset+1):(s_offset+samples_each)]
        pwess = p_weights ./ sum(p_weights)

        _ess = inv(sum(pwess .^ 2))

        if _ess <= N_t
            n_weights2 = p_weights .^ inv(γ)
            n_weights2 ./= sum(n_weights2)
        else
            n_weights2 = p_weights ./ sum(p_weights)
        end

        n_weights1 = p_weights ./ sum(p_weights)

        is_mean = sum(n_weights1 .* p_samples)

        is_mdiff = p_samples .- [is_mean for i = 1:samples_each]

        W = 1.0 - sum(n_weights1 .^ 2)
        is_cov1 = inv(W) .* sum(n_weights1 .* [i * i' for i in is_mdiff])
        is_cov2 = inv(W) .* sum(n_weights2 .* [i * i' for i in is_mdiff])

        Σ = (1.0 .- β) .* proposals[p_idx].Σ .+ β .* (1 - η) .* is_cov1 .+ β .* η .* is_cov2
        μ = (1.0 .- α) .* proposals[p_idx].μ .+ α .* is_mean

        proposals[p_idx] = MvNormal(μ, Σ)
    end
    return @dict samples weights
end
