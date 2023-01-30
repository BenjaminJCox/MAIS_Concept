using Distributions
using StatsBase
using DrWatson
using CairoMakie
using Random
using KernelDensity

include(srcdir("dmpmc.jl"))
include(srcdir("gapis.jl"))
include(srcdir("lais.jl"))
include(srcdir("rscais.jl"))
include(srcdir("cais.jl"))
include(srcdir("emscais.jl"))

include(srcdir("mixture_reduction.jl"))
include("p_helper.jl")

Random.seed!(123456789);

# target(x) = pdf(MvNormal([1,1], [1 0; 0 1]), x)

@inline function banana(x::Vector; b = 3.0, σ = 1.0)
    @assert length(x) > 2
    t1 = -x[1]^2 / (2σ^2)
    t2 = -((x[2] + b*(x[1]^2-σ^2))^2) / (2σ^2)
    t3 = -sum((x[3:end].^2) ./ (2σ^2))
    return exp(t1 + t2 + t3)
end

@inline target(x) = banana(x)

x_dim = 15
n_props = 25
n_iters = 200
prop_sigma = 1.0

ex_mse = Dict()

N = 1_000


dmpmc_props = init_proposals(x_dim, n_props, σ = prop_sigma)
dmpmc_init_adapt!(dmpmc_props, target; n_adapts = n_iters, samples_adapts = 100, global_adapt = false)
dmpmc_results = dmpmc_step!(dmpmc_props, target, N, global_adapt = true)

Zhat_dmpmc = mean(dmpmc_results[:weights])
is_mean_dmpmc = inv(Zhat_dmpmc) * mean(dmpmc_results[:weights] .* dmpmc_results[:samples])

is_mdiff_dmpmc = dmpmc_results[:samples] .- [is_mean_dmpmc for i = 1:(N*length(dmpmc_props))]

is_cov_dmpmc = inv(Zhat_dmpmc) * mean(dmpmc_results[:weights] .* [i * i' for i in is_mdiff_dmpmc])

ex_mse[:dmpmc] = mean(is_mean_dmpmc .^ 2)


# gapis_props = init_proposals(x_dim, n_props, σ = prop_sigma)
# for i = 1:n_iters
#     if i < 10
#         gapis_step!(gapis_props, target, 100, repulsion = true, λ = 0.1, scale_adapt = false)
#     else
#         gapis_step!(gapis_props, target, 100, repulsion = false, λ = 0.1, scale_adapt = false)
#     end
# end
# gapis_results = gapis_step!(gapis_props, target, N)
#
# Zhat_gapis = mean(gapis_results[:weights])
# is_mean_gapis = inv(Zhat_gapis) * mean(gapis_results[:weights] .* gapis_results[:samples])
#
# is_mdiff_gapis = gapis_results[:samples] .- [is_mean_gapis for i = 1:(N*length(gapis_props))]
#
# is_cov_gapis = inv(Zhat_gapis) * mean(gapis_results[:weights] .* [i * i' for i in is_mdiff_gapis])
#
# ex_mse[:gapis] = mean(is_mdiff_gapis .^ 2)


lais_props = init_proposals(x_dim, n_props, σ = prop_sigma)
for i = 1:n_iters
    lais_step!(lais_props, target, 5, repulsion = true, mcmc_steps = 5)
end
lais_results = lais_step!(lais_props, target, N, repulsion = true)

Zhat_lais = mean(lais_results[:weights])
is_mean_lais = inv(Zhat_lais) * mean(lais_results[:weights] .* lais_results[:samples])

is_mdiff_lais = lais_results[:samples] .- [is_mean_lais for i = 1:(N*length(lais_props))]

is_cov_lais = inv(Zhat_lais) * mean(lais_results[:weights] .* [i * i' for i in is_mdiff_lais])


ex_mse[:lais] = mean(is_mean_lais .^ 2)


rscais_props = init_proposals(x_dim, n_props, σ = prop_sigma)
for i = 1:n_iters
    rscais_step!(rscais_props, target, 100)
end
rscais_results = rscais_step!(rscais_props, target, N)

Zhat_rscais = mean(rscais_results[:weights])
is_mean_rscais = inv(Zhat_rscais) * mean(rscais_results[:weights] .* rscais_results[:samples])

is_mdiff_rscais = rscais_results[:samples] .- [is_mean_rscais for i = 1:(N*length(rscais_props))]

is_cov_rscais = inv(Zhat_rscais) * mean(rscais_results[:weights] .* [i * i' for i in is_mdiff_rscais])

ax5 = Axis(f1[2, 1], title = "Recursive Shrinkage AIS", xticks = LinearTicks(4), yticks = LinearTicks(4))
hist!(ax5, rscais_results[:weights] ./ sum(rscais_results[:weights]), bins = B, normalization = :pdf)
ylims!(0, nothing)
xlims!(0, nothing)
f1

ex_mse[:rscais] = sum(is_mean_rscais .^ 2)


# cais_props = init_proposals(x_dim, n_props, σ = prop_sigma)
# for i = 1:n_iters
#     cais_step!(cais_props, target, 100)
# end
# cais_results = cais_step!(cais_props, target, N)
#
# Zhat_cais = mean(cais_results[:weights])
# is_mean_cais = inv(Zhat_cais) * mean(cais_results[:weights] .* cais_results[:samples])
#
# is_mdiff_cais = cais_results[:samples] .- [is_mean_cais for i = 1:(N*length(cais_props))]
#
# is_cov_cais = inv(Zhat_cais) * mean(cais_results[:weights] .* [i * i' for i in is_mdiff_cais])
#
# ex_mse[:cais] = mean(is_mean_cais .^ 2)


rscais_gradual_props = init_proposals(x_dim, n_props, σ = prop_sigma)
rsc_mu = [i.μ for i in rscais_gradual_props]

for i = 1:n_iters
    # rscais_gradual_step!(rscais_gradual_props, target, 100, η = inv(i), β = 0.4 .* exp(-(i-1)/30))
    # kvk = rscais_gradual_step!(rscais_gradual_props, target, 100, η = inv(i), β = 0.1)
    kvk = rscais_gradual_step!(rscais_gradual_props, target, 140, η = inv(i), β = 0.4, α = 0.2)
    if i > 200
        # global rscais_gradual_props = merge_hellinger!(rscais_gradual_props, 0.2)
        # if i % 2 == 0
            # global rscais_gradual_props = merge_hellinger!(rscais_gradual_props, 0.4)
    #     elseif i % 5 == 0
    #         global rscais_gradual_props = mixture_culling_ess!(rscais_gradual_props, 100, kvk[:weights]; ν = 0.6)
        # end
    end
end
rscais_gradual_results = rscais_gradual_step!(rscais_gradual_props, target, N, η = inv(rs_iters + 1), β = 0.1)

Zhat_rscais_gradual = mean(rscais_gradual_results[:weights])
is_mean_rscais_gradual = inv(Zhat_rscais_gradual) * mean(rscais_gradual_results[:weights] .* rscais_gradual_results[:samples])

is_mdiff_rscais_gradual = rscais_gradual_results[:samples] .- [is_mean_rscais_gradual for i = 1:(N*length(rscais_gradual_props))]

is_cov_rscais_gradual =
    inv(Zhat_rscais_gradual) * mean(rscais_gradual_results[:weights] .* [i * i' for i in is_mdiff_rscais_gradual])

ex_mse[:rscais_gradual] = mean(is_mean_rscais_gradual .^ 2)


emscais_props = init_proposals(x_dim, n_props, σ = prop_sigma)

s_mu = [i.μ for i in emscais_props]

for i = 1:n_iters
    # emscais_step!(emscais_props, target, 100, η = inv(i), β = 0.4 .* exp(-(i-1)/30))
    # (ems_iters - i + 1)/ems_iters
    if i < 40
        kv = emscais_step!(emscais_props, target, 100, η = inv(i), β = 0.35, κ = inv(i), repulsion = true, α = 1.0, hmc_args = Dict(:K => 1e2))
        if i < 2
            global emscais_first = deepcopy(kv)
        end
    elseif i < 100
        kv = emscais_step!(emscais_props, target, 120, η = inv(i), β = 0.15, κ = inv(i), repulsion = true, α = 0.7, hmc_args = Dict(:K => 1.0))
    else
        kv = emscais_step!(emscais_props, target, 140, η = inv(i), β = 0.075, κ = inv(i), repulsion = false, α = 0.3)
    end
    # if i > 100
    #     if i % 2 == 0
    #         global emscais_props = merge_hellinger!(emscais_props, 0.2)
    #     elseif i % 5 == 0
    #         global emscais_props = split_ess!(emscais_props, 100, kv[:weights]; ν = 0.5)
    #     elseif i % 13
    #         global emscais_props = mixture_culling_ess!(emscais_props, 100, kv[:weights]; ν = 0.1)
    #     end
    # end
    # @info(i)
    # display(split_ess(emscais_props, 100, kv[:weights]; λ = 0.8, ν = 0.2))
end
emscais_results = emscais_step!(emscais_props, target, N, η = 0, β = 0.03, κ = 0, repulsion = false)

Zhat_emscais = mean(emscais_results[:weights])
is_mean_emscais = inv(Zhat_emscais) * mean(emscais_results[:weights] .* emscais_results[:samples])

is_mdiff_emscais = emscais_results[:samples] .- [is_mean_emscais for i = 1:(N*length(emscais_props))]

is_cov_emscais = inv(Zhat_emscais) * mean(emscais_results[:weights] .* [i * i' for i in is_mdiff_emscais])

ex_mse[:emscais] = mean(is_mean_emscais .^ 2)

display(ex_mse)
