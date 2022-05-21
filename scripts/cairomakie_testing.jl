using CairoMakie

xs = LinRange(-2, 2, 300)
ys = LinRange(-2, 5, 300)
zs = [(1.0 - x)^2 + 100.0 * (y - x^2)^2 for x in xs, y in ys]

f = Figure(resolution = (800, 800))
Axis(f[1, 1][1, 1])
xlims!(extrema(xs)...)
ylims!(extrema(ys)...)
co = contourf!(xs, ys, zs, levels = 10.0 .^ (-3:0.2:3), extendlow = :auto, extendhigh = :auto, colormap = :Spectral)
Colorbar(f[1, 1][1, 2], co, scale = log10)


Axis(f[1, 1][2, 1])
xlims!(0.7, 1.3)
ylims!(0.7, 1.3)

co = contourf!(xs, ys, zs, levels = 10.0 .^ (-5:0.2:1.6), extendlow = :auto, extendhigh = :auto, colormap = :Spectral)
Colorbar(f[1, 1][2, 2], co, scale = log10)

Axis(f[1, 2])
xlims!(0.7, 1.3)
ylims!(0.7, 1.3)

co = contourf!(xs, ys, zs, levels = 10.0 .^ (-5:0.2:1.6), extendlow = :auto, extendhigh = :auto, colormap = :Spectral)
Colorbar(f[1, 3], co, scale = log10)

f
