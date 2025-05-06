include("GenerateWeakOrder.jl")

using ArgParse
using JLD2
using UnPack

@kwdef struct Args
    M    :: UInt
    file :: String
    seed :: Union{Nothing, UInt}
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt; required = true; help = "Number of criteria")
        ("file"; required = true; help = "Partial sum file")
        (["--seed", "-s"]; arg_type = UInt; help = "Random seed")
    end

    return Args(; parse_args(s; as_symbols = true)...)
end

function main()
    @unpack M, file, seed = parse_commandline()

    Random.seed!(seed)

    S = load_object(file)::Vector{BigFloat}

    println(random_ranking(M, S))
    return 0
end

main()

# Base.ARGS = ["10", "src/julia/S/10.jld2"]
# @time main()
# @profview main()
# @code_warntype main()