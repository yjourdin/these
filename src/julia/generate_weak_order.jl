include("GenerateWeakOrder.jl")

using ArgParse
using JLD2
using UnPack

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt; required = true; help = "Number of criteria")
        ("Sfile"; required = true; help = "Partial sum file")
        (["--seed", "-s"]; arg_type = UInt; help = "Random seed")
    end

    return parse_args(s)
end

function main()
    @unpack M, Sfile, seed = parse_commandline()

    Random.seed!(seed)

    S = load_object(Sfile)

    println(random_ranking(M, S))
    return 0
end

main()

# Base.ARGS = ["10", "src/julia/S/10.txt"]
# @time main()
# @profview main()
# @code_warntype main()