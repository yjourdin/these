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
        "S"
            required = true
            help = "Partial sum file"
        "--seed", "-s"
            arg_type = UInt
            help = "Random seed"
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    Random.seed!(parsed_args["seed"])

    S = load_object(parsed_args["S"])

    println(random_ranking(parsed_args["m"], S))
end

main()

# ARGS=["10", "src/julia/S/10.txt"]
# @time main()
# @profview main()