module CheckUniformity

using ..Bit
using ..Posets: subset_decode
using ArgParse
using Chain
using DataStructures
using Distributions
using JLD2
using Random
using StatsBase
using UnPack

include("WE.jl")
include("successors.jl")
include("generate_weak_order_ext.jl")

@kwdef struct Args
    M    :: UInt
    N    :: UInt
    seed :: Union{Nothing, UInt}
end

function parse_commandline(args)
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt; required = true; help = "Number of criteria")
        ("N"; arg_type = UInt; required = true; help = "Number of samples")
        (["--seed", "-s"]; arg_type = UInt; help = "Random seed")
    end

    return Args(; parse_args(args, s; as_symbols = true)...)
end

function @main(args)
    @unpack M, N, seed = parse_commandline(args)

    Random.seed!(seed)

    file = jldopen(joinpath(dirname(@__DIR__), "WE", "$M.jld2"), "a")
    G = WE(file)

    K = G.nb_paths[1]
    @info "Nb paths : $K"

    c = counter(generate_weak_order_ext(G) for _ ∈ 1:N)

    T = (K / N) * sum(x^2 for x ∈ values(c)) - N

    println("Uniform : ", T < quantile(Chisq(K - 1), 0.95))
    println("P-value : ", ccdf(Chisq(K - 1), T))

    return 0
end

end # module CheckUniformity