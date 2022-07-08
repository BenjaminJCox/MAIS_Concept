using CairoMakie

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

function plot_marginals2d(proposals, target, xrange, yrange, samples, weights; title = "Marginals Plot")
    f3 = Figure(resolution = (1600, 1600))
    f3axtop = Axis(f3[1, 1], title = title)
    f3axmain = Axis(f3[2, 1])
    f3axright = Axis(f3[2, 2])

    linkyaxes!(f3axmain, f3axright)
    linkxaxes!(f3axmain, f3axtop)

    ylims!(f3axtop, low = 0)
    xlims!(f3axright, low = 0)

    _res = 200
    xs_main = LinRange(xrange..., _res)
    ys_main = LinRange(yrange..., _res)
    zs_main = [target([x, y]) for x in xs_main, y in ys_main]
    pl = heatmap!(f3axmain, xs_main, ys_main, zs_main, interpolate = true, colormap = :Oranges_9)

    mus = [i.μ for i in proposals]
    xs = [i[1] for i in mus]
    ys = [i[2] for i in mus]
    scatter!(f3axmain, xs, ys, color = "blue")

    a1 = quad_trap(vec(sum(zs_main, dims = 2)), xs_main)
    a2 = quad_trap(vec(sum(zs_main, dims = 1)), ys_main)

    t_xm = lines!(f3axtop, xs_main, vec(sum(zs_main, dims = 2)) ./ 0.5a1)
    t_ym = lines!(f3axright, vec(sum(zs_main, dims = 1)) ./ 0.5a2, ys_main)

    # t_xm = lines!(f3axtop, xs_main, vec(sum(zs_main, dims = 2)))
    # xlims!(f3axtop, extrema(xs_main))
    #
    # t_ym = lines!(f3axright, vec(sum(zs_main, dims = 1)), ys_main)
    # ylims!(f3axright, extrema(ys_main))

    hidedecorations!(f3axtop, grid = false)
    hidedecorations!(f3axright, grid = false)

    x_kde = kde(samples[:, 1], weights = weights)
    y_kde = kde(samples[:, 2], weights = weights)

    # a_xm = lines!(f2axtop, xs_main, vec(sum(zs_mixture, dims = 2)))
    a_xm = lines!(f3axtop, xs_main, pdf(x_kde, xs_main))
    xlims!(f3axtop, extrema(xs_main))

    # a_ym = lines!(f2axright, vec(sum(zs_mixture, dims = 1)), ys_main)
    a_ym = lines!(f3axright, pdf(y_kde, ys_main), ys_main)
    ylims!(f3axright, extrema(ys_main))

    # comp_mixture = MixtureModel(proposals)
    # zs_mixture = [pdf(comp_mixture, [x, y]) for x in xs_main, y in ys_main]
    #
    # a_xm = lines!(f3axtop, xs_main, vec(sum(zs_mixture, dims = 2)))
    # xlims!(f3axtop, extrema(xs_main))
    #
    # a_ym = lines!(f3axright, vec(sum(zs_mixture, dims = 1)), ys_main)
    # ylims!(f3axright, extrema(ys_main))

    leg = Legend(f3[1, 2],
    [t_xm, a_xm],
    ["True Marginal", "IS Marginal (SM)"])

    leg.tellheight = true

    colsize!(f3.layout, 1, Relative(mainsize))
    colsize!(f3.layout, 2, Relative(1-mainsize))
    rowsize!(f3.layout, 2, Relative(mainsize))
    rowsize!(f3.layout, 1, Relative(1-mainsize))

    f3
end
