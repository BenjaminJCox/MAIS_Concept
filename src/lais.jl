using LinearAlgebra
using Distributions
using DrWatson
using Zygote

include("weighting_schemes.jl")

@inline function hmc4lais!(
    θ,
    log_target;
    d_log_target = x -> gradient(k -> log_target(k), x)[1],
    n_iterations = 10,
    inv_M = Matrix(I(length(θ))),
    ϵ = 0.1,
    L = 10,
    repulsion = false,
    proposals = nothing,
    p_idx = p_idx
)
    # should be faster than calling AdvancedHMC routines as those have overhead to make the samples better
    # not needed here as samples themselves not of interest
    ϕ_dist = MvNormal(zero(θ), inv(inv_M))
    for i = 1:n_iterations
        θ′ = copy(θ)
        ϕ = rand(ϕ_dist)
        ϕ′ = copy(ϕ)
        if repulsion
            for leapfrog_step = 1:L
                ϕ′ .+= 0.5 .* ϵ .* (d_log_target(θ′) .+ coulomb_repulsion(θ′, proposals, p_idx))
                θ′ .+= ϵ .* inv_M * ϕ′
                ϕ′ .+= 0.5 .* ϵ .* (d_log_target(θ′) .+ coulomb_repulsion(θ′, proposals, p_idx))
            end
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

function lais_step!(
    proposals::Vector{MvNormal},
    target::Function,
    samples_each::Int;
    repulsion = true,
    mcmc_steps = 10,
)
    # we assume the proposals are normal, although we could modify this to not be the case: usual multivariate t may be better
    n_proposals = length(proposals)
    samples = Vector{Vector{Float64}}(undef, n_proposals * samples_each)
    wts = Vector{Float64}(undef, n_proposals * samples_each)
    Threads.@threads for p_idx = 1:n_proposals
        prop = proposals[p_idx]
        s_offset = (p_idx - 1) * samples_each
        for i = 1:samples_each
            samples[s_offset+i] = rand(prop)
            # wts[s_offset+i] = dm_weights(samples[s_offset+i], proposals, target)
        end
        wts[(s_offset+1):(s_offset+samples_each)] = dm_weights_new(samples[(s_offset+1):(s_offset+samples_each)], proposals, target)
    end
    weights = Weights(wts)
    Threads.@threads for p_idx = 1:n_proposals
        location = proposals[p_idx].μ
        new_location =
            hmc4lais!(location, x -> log(target(x)), repulsion = repulsion, proposals = proposals, n_iterations = mcmc_steps, p_idx = p_idx)
        new_scale = proposals[p_idx].Σ
        proposals[p_idx] = MvNormal(new_location, new_scale)
    end
    return @dict samples weights
end
