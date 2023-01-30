using LinearAlgebra
using Distributions
using InvertedIndices

@inline function dm_weights(x, proposals, target)
    return target(x) ./ (sum(pdf.(proposals, Ref(x))) ./ length(proposals))
end

function init_proposals(dim, n_proposals; lims = (-2, 2), σ = 1.0)
    proposals = Vector{MvNormal}(undef, n_proposals)
    plan = randomLHC(n_proposals, dim)
    plan = scaleLHC(plan, [lims for d in 1:dim])
    for p_idx = 1:n_proposals
        proposals[p_idx] = MvNormal(plan[p_idx, :], Matrix(σ .* I(dim)))
    end
    return proposals
end

@inline function coulomb_repulsion(θ, proposals, p_idx; K = 0.1, self_norm = false)
    # assuming mvnormal
    force = zero(θ)
    _p_det = sqrt(det(proposals[p_idx].Σ))
    for proposal in proposals[Not(p_idx)]
        separation = θ .- proposal.μ
        force .+= separation .* K .* _p_det .* sqrt(det(proposal.Σ)) ./ (norm(separation)^3)
    end
    if self_norm
        force ./= length(proposals)
    end
    return force
end
