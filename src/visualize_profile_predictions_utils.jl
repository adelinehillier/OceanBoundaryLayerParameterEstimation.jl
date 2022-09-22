function open_output_file(directory)
        isdir(directory) || mkpath(directory)
        file = joinpath(directory, "output.txt")
        touch(file)
        o = open(file, "w")
        return o
end

function horizontal_band!(ax, xlower, xupper, y; kwargs...)
    points = vcat([zip(xlower, y)...], reverse([zip(xupper, y)...]))
    poly!(ax, Point2f[points...]; kwargs...)
end

field_guide = Dict(
    :u => (
        units = "m/s",
        # name = "U-Velocity",
        name = "U"
    ),

    :v => (
        units = "m/s",
        # name = "V-Velocity",
        name = "V"
    ),

    :b => (
        units = "N/kg",
        name = "B",
        # name = "Buoyancy"
    ),

    :e => (
        units = "cm²/s²",
        # name = "TKE"
        name = "E"
    )
)

order_of_magnitude(num) = num == 0 ? 0 : Int(floor(log10(abs(num))))

function tostring(num)
    num == 0 && return "0"
    om = order_of_magnitude(num)
    num /= 10.0^om
    num = num%1 ≈ 0 ? Int(num) : round(num; digits=2)
    return "$(num)e$om"
end
