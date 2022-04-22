function open_output_file(directory)
        isdir(directory) || mkpath(directory)
        file = joinpath(directory, "output.txt")
        touch(file)
        o = open(file, "w")
        return o
end

field_guide = Dict(
    :u => (
        units = "m/s",
        name = "U-Velocity"
    ),

    :v => (
        units = "m/s",
        name = "V-Velocity"
    ),

    :b => (
        units = "N/kg",
        name = "Buoyancy"
    ),

    :e => (
        units = "cm²/s²",
        name = "TKE"
    )
)

order_of_magnitude(num) = Int(floor(log10(abs(num))))

function tostring(num)
    num == 0 && return "0"
    om = order_of_magnitude(num)
    num /= 10.0^om
    num = num%1 ≈ 0 ? Int(num) : round(num; digits=2)
    return "$(num)e$om"
end
