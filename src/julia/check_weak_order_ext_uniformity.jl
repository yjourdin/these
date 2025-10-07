include("GenerateWeakOrderExt.jl")

using ArgParse
using DataStructures
using Distributions
using JLD2
using UnPack

@kwdef struct Args
    file :: String
    N    :: UInt128
    seed :: Union{Nothing, UInt}
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("file"; required = true; help = "Input file")
        ("N"; arg_type = UInt128; required = true; help = "Number of samples")
        (["--seed", "-s"]; arg_type = UInt; help = "Random seed")
    end

    return Args(; parse_args(s; as_symbols = true)...)
end

function main()
    @unpack file, N, seed = parse_commandline()

    Random.seed!(seed)

    @unpack labels, nb_paths = file |> load |> WE

    K = nb_paths[1]
    @info "Nb paths : $K"

    c = counter(generate_weak_order_ext(labels, nb_paths) for _ ∈ 1:N)

    T = (K / N) * sum(x^2 for x ∈ values(c)) - N

    println("Uniform : ", T < quantile(Chisq(K - 1), 0.95))
    println("P-value : ", ccdf(Chisq(K - 1), T))
    return 0
end

main()

# Base.ARGS = ["3", "src/julia/WE/3.jld2", "10000"]
# @time main()
# @profview main()
# @code_warntype main()