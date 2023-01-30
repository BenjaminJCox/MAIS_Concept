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

Random.seed!(123456);

# target(x) = pdf(MvNormal([1,1], [1 0; 0 1]), x)

@inline function rapid_mvn_prec(x, mu, i_sigma, isq_d_sigma)
    k = size(mu, 1)

    t1 = (2π) .^ (-k ./ 2)
    t2 = isq_d_sigma
    lt3 = transpose(x .- mu) * i_sigma * (x .- mu)
    t3 = exp(-0.5 .* lt3)
    return (t1 .* t2 .* t3)
end

mu1 = [1.0, 1.0]
mu2 = [-3.0, 3.0]

sigma1 = [1.0 0.0; 0.0 1.0]
i_sigma1 = inv(sigma1)
isq_d_sigma1 = inv(sqrt(det(sigma1)))

sigma2 = [1.0 0.5; 0.5 2.0]
i_sigma2 = inv(sigma2)
isq_d_sigma2 = inv(sqrt(det(sigma2)))
@inline target(x) = 0.5 .* (rapid_mvn_prec(x, mu1, i_sigma1, isq_d_sigma1) .+ rapid_mvn_prec(x, mu2, i_sigma2, isq_d_sigma2))
target_dist = MixtureModel(MvNormal, [(mu1, sigma1), (mu2, sigma2)])

proposal = MvNormal([0.8, 1.5], [2 0; 0 2])

function perform_sir(tgt, prop, N)
    samples = Vector{Vector{Float64}}(undef, N)
    @simd for i = 1:N
        samples[i] = rand(prop)
    end
    pwts = pdf.(Ref(prop), samples)
    weights = Weights(tgt.(samples) ./ pwts)
    samples = sample(samples, weights, N; replace = true)
    weights = Weights([1 for i = 1:N])
    @dict samples weights
end

function perform_is(tgt, prop, N)
    samples = Vector{Vector{Float64}}(undef, N)
    @simd for i = 1:N
        samples[i] = rand(prop)
    end
    pwts = pdf.(Ref(prop), samples)
    weights = Weights(tgt.(samples) ./ pwts)
    return @dict samples weights
end

f1 = Figure(resolution = (1600, 800))

N = 1_000

sir_smp = perform_sir(target, proposal, N)
length(unique(sir_smp[:samples]))
is_mean_sir = mean(sir_smp[:samples])
is_cov_sir = cov(sir_smp[:samples])

is_res = perform_is(target, proposal, N)
Zhat = mean(is_res[:weights])
is_mean = inv(Zhat) * mean(is_res[:weights] .* is_res[:samples])

is_mdiff = is_res[:samples] .- [is_mean for i = 1:N]

is_cov = inv(Zhat) * mean(is_res[:weights] .* [i * i' for i in is_mdiff])

B = 12
ax1 = Axis(f1[1, 1], title = "Regular IS", xticks = LinearTicks(4), yticks = LinearTicks(4))
hist!(ax1, is_res[:weights] ./ sum(is_res[:weights]), bins = B, normalization = :pdf)
ylims!(0, nothing)
xlims!(0, nothing)
f1


dmpmc_props = init_proposals(2, 12, σ = 3.0)
dmpmc_init_adapt!(dmpmc_props, target; n_adapts = 30, samples_adapts = 100, global_adapt = false)
dmpmc_results = dmpmc_step!(dmpmc_props, target, N, global_adapt = true)

Zhat_dmpmc = mean(dmpmc_results[:weights])
is_mean_dmpmc = inv(Zhat_dmpmc) * mean(dmpmc_results[:weights] .* dmpmc_results[:samples])

is_mdiff_dmpmc = dmpmc_results[:samples] .- [is_mean_dmpmc for i = 1:(N*length(dmpmc_props))]

is_cov_dmpmc = inv(Zhat_dmpmc) * mean(dmpmc_results[:weights] .* [i * i' for i in is_mdiff_dmpmc])

ax2 = Axis(f1[1, 2], title = "Deterministic Mixture PMC", xticks = LinearTicks(4), yticks = LinearTicks(4))
hist!(ax2, dmpmc_results[:weights] ./ sum(dmpmc_results[:weights]), bins = B, normalization = :pdf)
ylims!(0, nothing)
xlims!(0, nothing)
f1


gapis_props = init_proposals(2, 12, σ = 3.0)
for i = 1:30
    if i < 10
        gapis_step!(gapis_props, target, 5, repulsion = false, λ = 0.1, scale_adapt = false)
    else
        gapis_step!(gapis_props, target, 5, repulsion = true, λ = 0.1, scale_adapt = false)
    end
end
gapis_results = gapis_step!(gapis_props, target, N)

Zhat_gapis = mean(gapis_results[:weights])
is_mean_gapis = inv(Zhat_gapis) * mean(gapis_results[:weights] .* gapis_results[:samples])

is_mdiff_gapis = gapis_results[:samples] .- [is_mean_gapis for i = 1:(N*length(gapis_props))]

is_cov_gapis = inv(Zhat_gapis) * mean(gapis_results[:weights] .* [i * i' for i in is_mdiff_gapis])

ax3 = Axis(f1[1, 3], title = "Gradient Adaptive Population IS", xticks = LinearTicks(4), yticks = LinearTicks(4))
hist!(ax3, gapis_results[:weights] ./ sum(gapis_results[:weights]), bins = B, normalization = :pdf)
ylims!(0, nothing)
xlims!(0, nothing)
f1


lais_props = init_proposals(2, 12, σ = 3.0)
for i = 1:30
    lais_step!(lais_props, target, 5, repulsion = true, mcmc_steps = 5)
end
lais_results = lais_step!(lais_props, target, N, repulsion = true)

Zhat_lais = mean(lais_results[:weights])
is_mean_lais = inv(Zhat_lais) * mean(lais_results[:weights] .* lais_results[:samples])

is_mdiff_lais = lais_results[:samples] .- [is_mean_lais for i = 1:(N*length(lais_props))]

is_cov_lais = inv(Zhat_lais) * mean(lais_results[:weights] .* [i * i' for i in is_mdiff_lais])

ax4 = Axis(f1[1, 4], title = "Layered AIS", xticks = LinearTicks(4), yticks = LinearTicks(4))
hist!(ax4, lais_results[:weights] ./ sum(lais_results[:weights]), bins = B, normalization = :pdf)
ylims!(0, nothing)
xlims!(0, nothing)
f1


rscais_props = init_proposals(2, 4, σ = 3.0)
for i = 1:30
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


cais_props = init_proposals(2, 4, σ = 3.0)
for i = 1:30
    cais_step!(cais_props, target, 100)
end
cais_results = cais_step!(cais_props, target, N)

Zhat_cais = mean(cais_results[:weights])
is_mean_cais = inv(Zhat_cais) * mean(cais_results[:weights] .* cais_results[:samples])

is_mdiff_cais = cais_results[:samples] .- [is_mean_cais for i = 1:(N*length(cais_props))]

is_cov_cais = inv(Zhat_cais) * mean(cais_results[:weights] .* [i * i' for i in is_mdiff_cais])

ax6 = Axis(f1[2, 2], title = "Covariance AIS", xticks = LinearTicks(4), yticks = LinearTicks(4))
hist!(ax6, cais_results[:weights] ./ sum(cais_results[:weights]), bins = B, normalization = :pdf)
ylims!(0, nothing)
xlims!(0, nothing)
f1


rscais_gradual_props = init_proposals(2, 12, σ = 3.0)
rsc_mu = [i.μ for i in rscais_gradual_props]

rs_iters = 400
for i = 1:rs_iters
    # rscais_gradual_step!(rscais_gradual_props, target, 100, η = inv(i), β = 0.4 .* exp(-(i-1)/30))
    # kvk = rscais_gradual_step!(rscais_gradual_props, target, 100, η = inv(i), β = 0.1)
    kvk = rscais_gradual_step!(rscais_gradual_props, target, 100, η = inv(i), β = 0.4, α = 0.2)
    if i > 200
        # global rscais_gradual_props = merge_hellinger!(rscais_gradual_props, 0.2)
        if i % 2 == 0
            global rscais_gradual_props = merge_hellinger!(rscais_gradual_props, 0.4)
        elseif i % 5 == 0
            global rscais_gradual_props = mixture_culling_ess!(rscais_gradual_props, 100, kvk[:weights]; ν = 0.6)
        end
    end
end
rscais_gradual_results = rscais_gradual_step!(rscais_gradual_props, target, N, η = inv(rs_iters + 1), β = 0.1)

Zhat_rscais_gradual = mean(rscais_gradual_results[:weights])
is_mean_rscais_gradual = inv(Zhat_rscais_gradual) * mean(rscais_gradual_results[:weights] .* rscais_gradual_results[:samples])

is_mdiff_rscais_gradual = rscais_gradual_results[:samples] .- [is_mean_rscais_gradual for i = 1:(N*length(rscais_gradual_props))]

is_cov_rscais_gradual =
    inv(Zhat_rscais_gradual) * mean(rscais_gradual_results[:weights] .* [i * i' for i in is_mdiff_rscais_gradual])

ax7 = Axis(f1[2, 3], title = "Gradual Recursive Shrinkage AIS", xticks = LinearTicks(4), yticks = LinearTicks(4))
hist!(ax7, rscais_gradual_results[:weights] ./ sum(rscais_gradual_results[:weights]), bins = B, normalization = :pdf)
ylims!(0, nothing)
xlims!(0, nothing)
f1


emscais_props = init_proposals(2, 12, σ = 3.0)

s_mu = [i.μ for i in emscais_props]

ems_iters = 150
for i = 1:ems_iters
    # emscais_step!(emscais_props, target, 100, η = inv(i), β = 0.4 .* exp(-(i-1)/30))
    # (ems_iters - i + 1)/ems_iters
    if i < 40
        kv = emscais_step!(emscais_props, target, 50, η = inv(i), β = 0.35, κ = inv(i), repulsion = true, α = 1.0, hmc_args = Dict(:K => 1e2))
        if i < 2
            global emscais_first = deepcopy(kv)
        end
    elseif i < 100
        kv = emscais_step!(emscais_props, target, 100, η = inv(i), β = 0.15, κ = inv(i), repulsion = true, α = 0.7, hmc_args = Dict(:K => 1.0))
    else
        kv = emscais_step!(emscais_props, target, 100, η = inv(i), β = 0.075, κ = inv(i), repulsion = false, α = 0.3)
    end
    if i > 100
        if i % 2 == 0
            global emscais_props = merge_hellinger!(emscais_props, 0.6)
        elseif i % 5 == 0
            global emscais_props = split_ess!(emscais_props, 100, kv[:weights]; ν = 0.5)
        else
            global emscais_props = mixture_culling_ess!(emscais_props, 100, kv[:weights]; ν = 0.5)
        end
    end
    # @info(i)
    # display(split_ess(emscais_props, 100, kv[:weights]; λ = 0.8, ν = 0.2))
end
emscais_results = emscais_step!(emscais_props, target, N, η = 0, β = 0.03, κ = 0, repulsion = false)

Zhat_emscais = mean(emscais_results[:weights])
is_mean_emscais = inv(Zhat_emscais) * mean(emscais_results[:weights] .* emscais_results[:samples])

is_mdiff_emscais = emscais_results[:samples] .- [is_mean_emscais for i = 1:(N*length(emscais_props))]

is_cov_emscais = inv(Zhat_emscais) * mean(emscais_results[:weights] .* [i * i' for i in is_mdiff_emscais])

ax8 = Axis(f1[2, 4], title = "Energy-Mean Shrinkage-Covariance AIS", xticks = LinearTicks(4), yticks = LinearTicks(4))
hist!(ax8, emscais_results[:weights] ./ sum(emscais_results[:weights]), bins = B, normalization = :pdf)
ylims!(0, nothing)
xlims!(0, nothing)

Label(f1[0, :], "Weight distribution", textsize = 20)

# linkaxes!(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8)
f1

function quad_trap(f_eval, xs)
    hs = diff(xs)
    int = 0
    N = length(xs)
    for k = 1:N-1
        xk = f_eval[k] + f_eval[k+1]
        int = int + hs[k] * xk
    end
    return int
end

f2 = Figure(resolution = (1600, 1600))
f2axtop = Axis(f2[1, 1], title = "EMSC AIS, $(ems_iters) iterations")
f2axmain = Axis(f2[2, 1])
f2axright = Axis(f2[2, 2])

linkyaxes!(f2axmain, f2axright)
linkxaxes!(f2axmain, f2axtop)

ylims!(f2axtop, low = 0)
xlims!(f2axright, low = 0)

_res = 200
xs_main = LinRange(-6.1, 4.1, _res)
ys_main = LinRange(-2.1, 6.1, _res)
zs_main = [target([x, y]) for x in xs_main, y in ys_main]
pl = heatmap!(f2axmain, xs_main, ys_main, zs_main, interpolate = true, colormap = :Oranges_9)

start_box = [[-2, -2], [-2, 2], [2, 2], [2, -2], [-2, -2]]
xs = [p[1] for p in start_box]
ys = [p[2] for p in start_box]
lines!(f2axmain, xs, ys, color = "red")

xs = [i[1] for i in s_mu]
ys = [i[2] for i in s_mu]
scatter!(f2axmain, xs, ys, color = "sienna", markersize = 5)

mus = [i.μ for i in emscais_props]
xs = [i[1] for i in mus]
ys = [i[2] for i in mus]
scatter!(f2axmain, xs, ys, color = "blue")

scatter!(f2axmain, Point2f.(sample(emscais_results[:samples], emscais_results[:weights], length(emscais_results[:samples]); replace = true)))

a1 = quad_trap(vec(sum(zs_main, dims = 2)), xs_main)
a2 = quad_trap(vec(sum(zs_main, dims = 1)), ys_main)

t_xm = lines!(f2axtop, xs_main, vec(sum(zs_main, dims = 2)) ./ 0.5a1)
t_ym = lines!(f2axright, vec(sum(zs_main, dims = 1)) ./ 0.5a2, ys_main)

hidedecorations!(f2axtop, grid = false)
hidedecorations!(f2axright, grid = false)

# comp_mixture = MixtureModel(emscais_props)
# zs_mixture = [pdf(comp_mixture, [x, y]) for x in xs_main, y in ys_main]

emscais_samples_demux = permutedims(hcat(emscais_results[:samples]...))
x_kde = kde(emscais_samples_demux[:, 1], weights = emscais_results[:weights])
y_kde = kde(emscais_samples_demux[:, 2], weights = emscais_results[:weights])

# a_xm = lines!(f2axtop, xs_main, vec(sum(zs_mixture, dims = 2)))
a_xm = lines!(f2axtop, xs_main, pdf(x_kde, xs_main))
xlims!(f2axtop, extrema(xs_main))


# a_ym = lines!(f2axright, vec(sum(zs_mixture, dims = 1)), ys_main)
a_ym = lines!(f2axright, pdf(y_kde, ys_main), ys_main)
ylims!(f2axright, extrema(ys_main))

leg = Legend(f2[1, 2], [t_xm, a_xm], ["True Marginal", "IS Marginal (SM)"])

leg.tellheight = true

mainsize = 9 / 10

colsize!(f2.layout, 1, Relative(mainsize))
colsize!(f2.layout, 2, Relative(1 - mainsize))
rowsize!(f2.layout, 2, Relative(mainsize))
rowsize!(f2.layout, 1, Relative(1 - mainsize))

f2


f3 = Figure(resolution = (1600, 1600))
f3axtop = Axis(f3[1, 1], title = "Recursive Shrinkage Covariance AIS, $(rs_iters) iterations")
f3axmain = Axis(f3[2, 1])
f3axright = Axis(f3[2, 2])

linkyaxes!(f3axmain, f3axright)
linkxaxes!(f3axmain, f3axtop)

ylims!(f3axtop, low = 0)
xlims!(f3axright, low = 0)

_res = 200
xs_main = LinRange(-6.1, 4.1, _res)
ys_main = LinRange(-2.1, 6.1, _res)
zs_main = [target([x, y]) for x in xs_main, y in ys_main]
pl = heatmap!(f3axmain, xs_main, ys_main, zs_main, interpolate = true, colormap = :Oranges_9)

start_box = [[-2, -2], [-2, 2], [2, 2], [2, -2], [-2, -2]]
xs = [p[1] for p in start_box]
ys = [p[2] for p in start_box]
lines!(f3axmain, xs, ys, color = "red")

xs = [i[1] for i in rsc_mu]
ys = [i[2] for i in rsc_mu]
scatter!(f3axmain, xs, ys, color = "sienna", markersize = 5)

mus = [i.μ for i in rscais_gradual_props]
xs = [i[1] for i in mus]
ys = [i[2] for i in mus]
scatter!(f3axmain, xs, ys, color = "blue")

# t_xm = lines!(f3axtop, xs_main, vec(sum(zs_main, dims = 2)))
# xlims!(f3axtop, extrema(xs_main))
#
# t_ym = lines!(f3axright, vec(sum(zs_main, dims = 1)), ys_main)
# ylims!(f3axright, extrema(ys_main))

a1 = quad_trap(vec(sum(zs_main, dims = 2)), xs_main)
a2 = quad_trap(vec(sum(zs_main, dims = 1)), ys_main)

t_xm = lines!(f3axtop, xs_main, vec(sum(zs_main, dims = 2)) ./ 0.5a1)
t_ym = lines!(f3axright, vec(sum(zs_main, dims = 1)) ./ 0.5a2, ys_main)

hidedecorations!(f3axtop, grid = false)
hidedecorations!(f3axright, grid = false)

# comp_mixture = MixtureModel(rscais_gradual_props)
# zs_mixture = [pdf(comp_mixture, [x, y]) for x in xs_main, y in ys_main]
#
# a_xm = lines!(f3axtop, xs_main, vec(sum(zs_mixture, dims = 2)))
# xlims!(f3axtop, extrema(xs_main))
#
# a_ym = lines!(f3axright, vec(sum(zs_mixture, dims = 1)), ys_main)
# ylims!(f3axright, extrema(ys_main))

rscais_samples_demux = permutedims(hcat(rscais_results[:samples]...))
x_kde = kde(rscais_samples_demux[:, 1], weights = rscais_results[:weights])
y_kde = kde(rscais_samples_demux[:, 2], weights = rscais_results[:weights])

# a_xm = lines!(f2axtop, xs_main, vec(sum(zs_mixture, dims = 2)))
a_xm = lines!(f3axtop, xs_main, pdf(x_kde, xs_main))
xlims!(f3axtop, extrema(xs_main))

# a_ym = lines!(f2axright, vec(sum(zs_mixture, dims = 1)), ys_main)
a_ym = lines!(f3axright, pdf(y_kde, ys_main), ys_main)
ylims!(f3axright, extrema(ys_main))

leg = Legend(f3[1, 2], [t_xm, a_xm], ["True Marginal", "IS Marginal (SM)"])

leg.tellheight = true

colsize!(f3.layout, 1, Relative(mainsize))
colsize!(f3.layout, 2, Relative(1 - mainsize))
rowsize!(f3.layout, 2, Relative(mainsize))
rowsize!(f3.layout, 1, Relative(1 - mainsize))

f3

# f2ax2 = Axis(f2[1, 2], title = "Gradual RSC AIS, $(rs_iters) iterations")
# _res = 200
# xs = LinRange(-6.1, 4.1, _res)
# ys = LinRange(-2.1, 6.1, _res)
# zs = [target([x, y]) for x in xs, y in ys]
# pl = heatmap!(f2ax2, xs, ys, zs, interpolate = true, colormap = :cividis)
#
# start_box = [[-2, -2], [-2, 2], [2, 2], [2, -2], [-2, -2]]
# xs = [p[1] for p in start_box]
# ys = [p[2] for p in start_box]
# lines!(f2ax2, xs, ys, color = "red")
#
# xs = [i[1] for i in rsc_mu]
# ys = [i[2] for i in rsc_mu]
# scatter!(f2ax2, xs, ys, color = "sienna", markersize = 5)
#
# mus = [i.μ for i in rscais_gradual_props]
# xs = [i[1] for i in mus]
# ys = [i[2] for i in mus]
# scatter!(f2ax2, xs, ys, color = "red")
#
# f2

f4 = plot_marginals2d(dmpmc_props, target, (-6.1, 4.1), (-2.1, 6.1), permutedims(hcat(dmpmc_results[:samples]...)), dmpmc_results[:weights], title = "DMPMC, 30 iterations")


# save(plotsdir("is_weight_degen.pdf"), f2)

f2
