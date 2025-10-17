module GenerateLinext

include("../../Bit.jl")
using .Bit: decode
include("../../Posets.jl")
using .Posets:
    above,
    below,
    just_above,
    just_below,
    nv,
    maximals,
    minimals,
    remove_vertex!,
    subset_lattice
using ArgParse
using Chain
using Random
using StatsBase
using UnPack

include("isolated.jl")
include("cardinality.jl")
include("probabilities.jl")
include("select_extremal.jl")
include("generate_linext.jl")

@kwdef struct Args
    M    :: UInt
    seed :: Union{Nothing, UInt}
end

function parse_commandline(args)
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt; required = true; help = "Number of criteria")
        (["--seed", "-s"]; arg_type = UInt; help = "Random seed")
    end

    return Args(; parse_args(args, s; as_symbols = true)...)
end

function @main(args)
    @unpack M, seed = parse_commandline(args)

    Random.seed!(seed)

    M |> subset_lattice |> generate_linext! |> println

    return 0
end

end # module GenerateLinext
