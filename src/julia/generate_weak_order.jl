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
        "Sfile"
            required = true
            help = "Partial sum file"
        "--seed", "-s"
            arg_type = UInt
            help = "Random seed"
    end

    parsed_args = parse_args(s)

    return (
        parsed_args["M"]::UInt,
        parsed_args["Sfile"]::String,
        parsed_args["seed"]::Union{UInt, Nothing}
    )
end

function main()
    (M, Sfile, seed) = parse_commandline()

    Random.seed!(seed)

    S = load_object(Sfile)

    println(random_ranking(M, S))
end

main()

# Base.ARGS = ["10", "src/julia/S/10.txt"]
# @time main()
# @profview main()