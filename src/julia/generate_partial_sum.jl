include("GenerateWeakOrder.jl")

using ArgParse
using JLD2


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "M"
            arg_type = UInt
            required = true
            help = "Number of criteria"
        "--output", "-o"
            help = "Output file"
    end

    parsed_args = parse_args(s)

    return (
        parsed_args["M"]::UInt,
        parsed_args["output"]::Union{Nothing, String}
    )
end

function main()
    (M, output) = parse_commandline()

    S = generate_partial_sum(M)

    if isnothing(output)
        println(S)
    else
        save_object(output, S)
    end
end

main()

# Base.ARGS = ["10"]
# @time main()
# @profview main()