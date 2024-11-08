include("GenerateLinext.jl")

using ArgParse


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "M"
            arg_type = UInt
            required = true
            help = "Number of criteria"
        "--seed", "-s"
            arg_type = UInt
            help = "Random seed"
    end

    parsed_args = parse_args(s)

    return (
        parsed_args["M"]::UInt,
        parsed_args["seed"]::Union{UInt, Nothing}
    )
end

function main()
    (M, seed) = parse_commandline()

    Random.seed!(seed)
    
    println(generate_linext(BooleanLattice(Int(M))))
end

main()

# Base.ARGS = ["11"]
# @time main()
# @profview main()