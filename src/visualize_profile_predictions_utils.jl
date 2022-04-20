function open_output_file(directory)
        isdir(directory) || mkpath(directory)
        file = joinpath(directory, "output.txt")
        touch(file)
        o = open(file, "w")
        return o
end

function writeout(o, name, loss, params)
        param_vect = [params...]
        loss_value = loss(params)
        write(o, "----------- \n")
        write(o, "$(name) \n")
        write(o, "Parameters: $(param_vect) \n")
        write(o, "Loss: $(loss_value) \n")
        saveplot(params, name, loss)
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
