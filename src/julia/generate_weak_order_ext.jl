include("GenerateWeakOrderExt.jl")

using ArgParse
using JLD2
using UnPack

@kwdef struct Args
    file::String
    seed::Union{Nothing, UInt}
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("file"; required = true; help = "Input file")
        (["--seed", "-s"]; arg_type = UInt; help = "Random seed")
    end

    return Args(; parse_args(s; as_symbols = true)...)
end

function main()
    @unpack file, seed = parse_commandline()

    Random.seed!(seed)

    @unpack labels, nb_paths = file |> load |> WE

    println(generate_weak_order_ext(labels, nb_paths))
    return 0
end

main()

# Base.ARGS = ["src/julia/WE/2.jld2"]
# @time main()
# @profview main()
# @code_warntype main()