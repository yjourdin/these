include("GenerateWeakOrder.jl")

using ArgParse
using JLD2


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "m"
            arg_type = UInt
            required = true
            help = "Number of criteria"
        "--output", "-o"
            help = "Output file"
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    S = generate_partial_sum(parsed_args["m"])

    if isnothing(parsed_args["output"])
        println(S)
    else
        save_object(parsed_args["output"], S)
    end
end

main()

# ARGS=["10"]
# @time main()
# @profview main()