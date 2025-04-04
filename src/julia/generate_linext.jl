include("GenerateLinext.jl")

using ArgParse
using UnPack

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt; required = true; help = "Number of criteria")
        (["--seed", "-s"]; arg_type = UInt; help = "Random seed")
    end

    return parse_args(s)
end

function main()
    @unpack M, seed = parse_commandline()

    Random.seed!(seed)

    println(generate_linext(BooleanLattice(Int(M))))
    return 0
end

main()

# Base.ARGS = ["2"]
# @time main()
# @profview main()
# @code_warntype main()