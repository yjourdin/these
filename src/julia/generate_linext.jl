include("GenerateLinext.jl")

using ArgParse


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "m"
            arg_type = UInt
            required = true
            help = "Number of criteria"
        "--seed", "-s"
            arg_type = UInt
            help = "Random seed"
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    Random.seed!(parsed_args["seed"])
    
    println(generate_linext(BooleanLattice(Int(parsed_args["m"]))))
end

main()

# ARGS=["11"]
# @time main()
# @profview main()