using Distributions
using StatsBase
using DrWatson
using CairoMakie
using Random
using KernelDensity
using Zygote
using JLD2

include(srcdir("dmpmc.jl"))
include(srcdir("gapis.jl"))
include(srcdir("lais.jl"))
include(srcdir("rscais.jl"))
include(srcdir("cais.jl"))
include(srcdir("emscais.jl"))

include(srcdir("mixture_reduction.jl"))
include("p_helper.jl")

# Random.seed!(123456789);

# target(x) = pdf(MvNormal([1,1], [1 0; 0 1]), x)

@inline function banana(x; b = 3.0, σ = 1.0)
    @assert length(x) > 2
    t1 = -x[1]^2 / (2σ^2)
    t2 = -((x[2] + b*(x[1]^2-σ^2))^2) / (2σ^2)
    t3 = -sum((x[3:end].^2) ./ (2σ^2))
    return exp(t1 + t2 + t3)
end

@inline function l_banana(x; b = 3.0, σ = 1.0)
    @assert length(x) > 2
    t1 = -x[1]^2 / (2σ^2)
    t2 = -((x[2] + b*(x[1]^2-σ^2))^2) / (2σ^2)
    t3 = -sum((x[3:end].^2) ./ (2σ^2))
    return t1 + t2 + t3
end

@inline function g_l_banana(x; b = 3.0, σ = 1.0)
    @assert length(x) > 2
    _rv = zero(x)
    _rv[3:end] = -x[3:end] ./ (σ^2)
    _rv[2] = - (b * (x[1]^2 - σ^2) + x[2])/(σ^2)
    _rv[1] = -(x[1] ./ (σ^2)) -(-2*b^2*σ^2*x[1] + 2*b^2*x[1]^3 + 2*b*x[1]*x[2])/(σ^2)
    return _rv
end


@inline target(x) = banana(x)
@inline log_target(x) = l_banana(x)
@inline d_ltarget(x) = gradient(k -> log_target(k), x)[1]

# x_dim = 15
n_props = 32
n_iters = 150
prop_sigma = 1.0

n_runs = 10

ex_mse = Dict()

N = 200

# _dims = [3, 5, 7, 10]
_dims = [15, 20, 25, 30]
# _dims = [3, 5, 10, 15, 20, 25, 30]
_results = Vector{Dict}(undef, length(_dims))

for _idx in 1:length(_dims)
    _results[_idx] = Dict()

    _results[_idx][:dmpmc] = zeros(n_runs)
    _results[_idx][:rscais_gradual] = zeros(n_runs)
    _results[_idx][:emscais] = zeros(n_runs)
end

# Threads.@threads for run in 1:n_runs

function run_test!(;n_runs = n_runs, _dims = _dims, _results = _results, N = N, n_props = n_props, prop_sigma = prop_sigma)
    Threads.@threads for run in 1:n_runs
        @info("BEGINNING RUN $run")
        for _idx in 1:length(_dims)
            dim = _dims[_idx]

            # @info(dim)

            dmpmc_props = init_proposals(dim, n_props, σ = prop_sigma)

            dmpmc_samples = Vector{Vector{Float64}}(undef, n_props * n_iters * N)
            dmpmc_weights = Vector{Float64}(undef, n_props * n_iters * N)

            _SPI = n_props * N

            t_samples = Vector{Vector{Float64}}(undef, _SPI)
            t_weights = Vector{Float64}(undef, _SPI)

            for i in 1:_SPI
                t_weights[i] = 0.0
                t_samples[i] = zeros(dim)
            end

            for i = 1:n_iters
                dmpmc_step!!(dmpmc_props, target, N, global_adapt = true, samples = t_samples, weights = t_weights)
                _offset = (n_props * N * (i-1))+1
                dmpmc_samples[_offset:_offset+_SPI-1] = t_samples
                dmpmc_weights[_offset:_offset+_SPI-1] = t_weights
            end

            # @info(size(dmpmc_samples))
            # dmpmc_samples = dmpmc_results[:samples]
            # dmpmc_weights = dmpmc_results[:weights]

            Zhat_dmpmc = mean(dmpmc_weights)
            is_mean_dmpmc = inv(Zhat_dmpmc) * mean(dmpmc_weights .* dmpmc_samples)
            is_mdiff_dmpmc = dmpmc_samples .- [is_mean_dmpmc for i = 1:(n_iters*_SPI)]
            # is_cov_dmpmc = inv(Zhat_dmpmc) * mean(dmpmc_weights .* [i * i' for i in is_mdiff_dmpmc])
            _results[_idx][:dmpmc][run] = mean(is_mean_dmpmc .^ 2)

            dmpmc_samples = nothing
            dmpmc_weights = nothing

            @info("GR-DMPMC RUN $run DIM $dim Done")

            GC.gc()


            rscais_gradual_props = init_proposals(dim, n_props, σ = prop_sigma)
            rsc_mu = [i.μ for i in rscais_gradual_props]

            rscais_samples = Vector{Vector{Float64}}(undef, n_props * n_iters * N)
            rscais_weights = Vector{Float64}(undef, n_props * n_iters * N)

            for i = 1:n_iters
                betaget = 0.05 + (n_iters-i) * (0.3 / n_iters)
                rscais_results = rscais_gradual_step!!(rscais_gradual_props, target, N, η = inv(i), β = betaget, α = 1.0, samples = t_samples, weights = t_weights)
                _offset = (n_props * N * (i-1))+1
                rscais_samples[_offset:_offset+_SPI-1] = t_samples
                rscais_weights[_offset:_offset+_SPI-1] = t_weights
            end

            Zhat_rscais_gradual = mean(rscais_weights)
            is_mean_rscais_gradual = inv(Zhat_rscais_gradual) * mean(rscais_weights .* rscais_samples)
            is_mdiff_rscais_gradual = rscais_samples .- [is_mean_rscais_gradual for i = 1:(n_iters*_SPI)]
            # is_cov_rscais_gradual =
                # inv(Zhat_rscais_gradual) * mean(rscais_weights .* [i * i' for i in is_mdiff_rscais_gradual])
            _results[_idx][:rscais_gradual][run] = mean(is_mean_rscais_gradual .^ 2)

            rscais_samples = nothing
            rscais_weights = nothing

            @info("RSCAIS RUN $run DIM $dim Done")

            GC.gc()


            emscais_props = init_proposals(dim, n_props, σ = prop_sigma)
            s_mu = [i.μ for i in emscais_props]

            emscais_samples = Vector{Vector{Float64}}(undef, n_props * n_iters * N)
            emscais_weights = Vector{Float64}(undef, n_props * n_iters * N)

            for i = 1:n_iters
                betaget = 0.05 + (n_iters-i) * (0.3 / n_iters)
                alphaget = (0.1 + (n_iters-i) * (0.8 / n_iters))^(1.5)
                # kappaget =  0.05 + (n_iters-i) * (0.2 / n_iters)
                kappaget = inv(i)
                if i < 40
                    emscais_results = emscais_step!!(emscais_props, target, N, η = inv(i), β = betaget, κ = kappaget, repulsion = true, α = alphaget, hmc_args = Dict(:K => 10.0, :d_log_target => g_l_banana, :ϵ => 0.05, :L => 5), samples = t_samples, weights = t_weights)
                elseif i < 100
                    emscais_results = emscais_step!!(emscais_props, target, N, η = inv(i), β = betaget, κ = kappaget, repulsion = true, α = alphaget, hmc_args = Dict(:K => 1.0, :d_log_target => g_l_banana, :ϵ => 0.05, :L => 5), samples = t_samples, weights = t_weights)
                else
                    emscais_results = emscais_step!!(emscais_props, target, N, η = inv(i), β = betaget, κ = 0, repulsion = false, α = alphaget, hmc_args = Dict(:K => 0.1, :d_log_target => g_l_banana, :ϵ => 0.05, :L => 2), samples = t_samples, weights = t_weights)
                end
                _offset = (n_props * N * (i-1))+1
                emscais_samples[_offset:_offset+_SPI-1] = t_samples
                emscais_weights[_offset:_offset+_SPI-1] = t_weights
            end

            # @info(size(emscais_samples))
            Zhat_emscais = mean(emscais_weights)
            is_mean_emscais = inv(Zhat_emscais) * mean(emscais_weights .* emscais_samples)
            is_mdiff_emscais = emscais_samples .- [is_mean_emscais for i = 1:(n_iters*_SPI)]
            # is_cov_emscais = inv(Zhat_emscais) * mean(emscais_weights .* [i * i' for i in is_mdiff_emscais])
            _results[_idx][:emscais][run] = mean(is_mean_emscais .^ 2)
            @info("EMSCAIS RUN $run DIM $dim Done")

            emscais_samples = nothing
            emscais_weights = nothing

            GC.gc()
        end
    end
    @info("Done Estimating!")
    return _results
end

run_test!()

dkr = deepcopy(_results)

for _idx in 1:length(_dims)
    for k in keys(dkr[_idx])
        dkr[_idx][k] = mean(dkr[_idx][k])
    end
end

_time = Integer(round(time()))
#
jldsave("banana_results_$_time.jld2"; results = _results, dims = _dims)

# display(ex_mse)

# display(emscais_dm)

function cull_diagonal(a::Matrix)
    sz = LinearIndices(a)
    n = size(a, 1)
    k = [sz[i,i] for i in 1:n]
    b = collect(vec(a'))
    deleteat!(b, k)
    collect(reshape(b, n - 1, n)')
end
# emscais_dm_c = cull_diagonal(emscais_dm)
# v_emscais_dm = vec(emscais_dm_c)
# hist(v_emscais_dm, axis = (title = "$x_dim dimensional target"))

dmpmc_vec = [x[:dmpmc] for x in dkr]
rscais_vec = [x[:rscais_gradual] for x in dkr]
emscais_vec = [x[:emscais] for x in dkr]

f = Figure()
ax = Axis(f[1, 1], xlabel = "target dimension", ylabel = "MSE", xticks = [0, _dims...], yscale = log10)
xlims!(ax, [0, maximum(_dims)*1.1])
dm = scatterlines!(f[1,1], _dims, dmpmc_vec)
rs = scatterlines!(f[1,1], _dims, rscais_vec)
em = scatterlines!(f[1,1], _dims, emscais_vec)
Legend(f[1,2],
    [dm, rs, em],
    ["DMPMC", "RSCAIS", "EMSCAIS"])
f
