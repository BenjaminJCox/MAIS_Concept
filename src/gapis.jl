using LinearAlgebra
using Distributions
using DrWatson
using Zygote

include("weighting_schemes.jl")

function gapis_step!(
    proposals::Vector{MvNormal},
    target::Function,
    samples_each::Int;
    λ = 0.05,
    dlπ = x -> gradient(k -> log(target(k)), x),
    Hπ = x -> hessian(k -> -log(target(k)), x),
    repulsion = false,
    scale_adapt = true
)
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
    for p_idx = 1:n_proposals
        location = proposals[p_idx].μ
        scale = proposals[p_idx].Σ
        if repulsion
            new_location = location .+ (λ .* (dlπ(location)[1] .+ coulomb_repulsion(location, proposals, p_idx)))
        else
            new_location = location .+ (λ .* dlπ(location)[1])
        end

        if scale_adapt
            new_scale = inv(Hπ(location))
            new_scale .+= new_scale'
            new_scale ./= 2

            new_scale = (1-λ) .* scale .+ λ .* new_scale
            new_scale .+= new_scale'
            new_scale ./= 2
        else
            new_scale = scale
        end
        proposals[p_idx] = MvNormal(new_location, new_scale)
    end
    return @dict samples weights
end
