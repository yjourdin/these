include("GenerateLinext.jl")

using ArgParse
using UnPack

@kwdef struct Args
    M    :: UInt
    seed :: Union{Nothing, UInt}
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt; required = true; help = "Number of criteria")
        (["--seed", "-s"]; arg_type = UInt; help = "Random seed")
    end

    return Args(; parse_args(s; as_symbols = true)...)
end

function main()
    @unpack M, seed = parse_commandline()

    Random.seed!(seed)

    println(generate_linext!(subset_lattice(M)))
    return 0
end

main()

# Base.ARGS = ["11"]
# @time main()
# @profview main()
# @code_warntype main()