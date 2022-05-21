using LinearAlgebra
using Distributions
using DrWatson
using Zygote

include("weighting_schemes.jl")

@inline function hmc4emscais!(
    θ,
    log_target;
    d_log_target = x -> gradient(k -> log_target(k), x)[1],
    n_iterations = 10,
    inv_M = Matrix(I(length(θ))),
    ϵ = 0.1,
    L = 10,
    repulsion = false,
    proposals = nothing,
    p_idx = p_idx,
)
    # should be faster than calling AdvancedHMC routines as those have overhead to make the samples better
    # not needed here as samples themselves not of interest
    ϕ_dist = MvNormal(zero(θ), inv(inv_M))
    for i = 1:n_iterations
        θ′ = copy(θ)
        ϕ = rand(ϕ_dist)
        ϕ′ = copy(ϕ)
        if repulsion == true
            for leapfrog_step = 1:L
                # @info("log_target", d_log_target(θ′))
                # @info("repulsion", emscais_coulomb_repulsion(θ′, proposals, p_idx))
                ϕ′ .+= 0.5 .* ϵ .* (d_log_target(θ′) .+ coulomb_repulsion(θ′, proposals, p_idx))
                θ′ .+= ϵ .* inv_M * ϕ′
                ϕ′ .+= 0.5 .* ϵ .* (d_log_target(θ′) .+ coulomb_repulsion(θ′, proposals, p_idx))
            end
            # @info("log_target", d_log_target(θ′))
            # @info("repulsion", emscais_coulomb_repulsion(θ′, proposals, p_idx))
        else
            for leapfrog_step = 1:L
                ϕ′ .+= 0.5 .* ϵ .* d_log_target(θ′)
                θ′ .+= ϵ .* inv_M * ϕ′
                ϕ′ .+= 0.5 .* ϵ .* d_log_target(θ′)
            end
        end
        mha = log_target(θ′) + logpdf(ϕ_dist, ϕ′) - log_target(θ) - logpdf(ϕ_dist, ϕ)
        if log(rand()) < mha
            θ = θ′
        end
    end
    return θ
end

function emscais_step!(
    proposals::Vector{MvNormal},
    target::Function,
    samples_each::Int;
    α = 0.1,
    β = 0.1,
    η = 0.1,
    κ = 0.1,
    γ = 3,
    N_t = 0.1 * samples_each,
    repulsion = true,
    mcmc_steps = 1,
)
    # Energy-Mean Shrinkage-Covariance (Multiple) Adaptive Importance Sampling
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

        _ess = inv(sum(p_weights .^ 2))

        if _ess <= N_t
            n_weights2 = p_weights .^ inv(γ)
            n_weights2 ./= sum(n_weights2)
        else
            n_weights2 = p_weights ./ sum(p_weights)
        end

        n_weights1 = p_weights ./ sum(p_weights)

        is_mean = sum(n_weights1 .* p_samples)

        if κ > 0
            sampled_mean = hmc4emscais!(
                proposals[p_idx].μ,
                x -> log(target(x)),
                repulsion = repulsion,
                proposals = proposals,
                n_iterations = mcmc_steps,
                p_idx = p_idx,
            )
        else
            sampled_mean = zero(is_mean)
        end

        is_mdiff = p_samples .- [is_mean for i = 1:samples_each]

        W = 1.0 - sum(n_weights1 .^ 2)
        is_cov1 = inv(W) .* sum(n_weights1 .* [i * i' for i in is_mdiff])
        is_cov2 = inv(W) .* sum(n_weights2 .* [i * i' for i in is_mdiff])

        Σ = (1.0 .- β) .* proposals[p_idx].Σ .+ β .* (1 - η) .* is_cov1 .+ β .* η .* is_cov2
        μ = (1.0 .- α) .* proposals[p_idx].μ .+ α .* (1 - κ) .* is_mean .+ α .* κ .* sampled_mean

        proposals[p_idx] = MvNormal(μ, Σ)
    end
    return @dict samples weights
end
