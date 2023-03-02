using LinearAlgebra
using Distributions
using Random

function est_sqhellinger(p_dist, q_dist, n_samples; sample_dist = MixtureModel([p_dist, q_dist]))
    @assert length(p_dist) == length(q_dist) == length(sample_dist)

    samples = rand(sample_dist, n_samples)

    p_term = pdf(p_dist, samples)
    q_term = pdf(q_dist, samples)

    # g_term = (sqrt.(p_term) .- sqrt.(q_term)).^2
    g_term = sqrt.(p_term .* q_term)

    weights = g_term ./ pdf(sample_dist, samples)

    sqhell_est = 1.0 - mean(weights)

    return sqhell_est
end
