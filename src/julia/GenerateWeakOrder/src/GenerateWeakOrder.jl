module GenerateWeakOrder

using ArgParse
using JLD2
using Memoization
using Random
using StatsBase
using UnPack

include("number_weak_orders.jl")
include("generate_partial_sum.jl")
include("random.jl")

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

    file = joinpath(dirname(@__DIR__), "S", "$M.jld2")
    isfile(file) || save_object(file, generate_partial_sum(M))
    S = load_object(file)::PartialSum

    println(random_ranking(M, S))

    return 0
end

end # module GenerateWeakOrder
