module GenerateWeakOrderExt

include("../../Bit.jl")
using .Bit
include("../../Posets.jl")
using .Posets: subset_lattice, subset_decode
using ArgParse
using Chain
using JLD2
using Random
using StatsBase
using UnPack

include("bit_poset.jl")
include("all_weak_3.jl")
include("WE.jl")
include("successors.jl")
include("generate_WE.jl")
include("generate_weak_order_ext.jl")

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

    file = jldopen(joinpath(dirname(@__DIR__), "WE", "$M.jld2"), "a")
        if isempty(file)
            @unpack labels, nb_paths = M |> subset_lattice |> generate_WE
            @pack! file = labels, nb_paths
        end

        file |> WE |> generate_weak_order_ext |> println
    close(file)

    return 0
end

include("CheckUniformity.jl")

end # module GenerateWeakOrderExt
