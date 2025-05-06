include("GenerateWeakOrderExt.jl")

using ArgParse
using DataStructures
using Distributions
using JLD2

@kwdef struct Args
    M    :: UInt128
    file :: String
    N    :: UInt128
    seed :: Union{Nothing, UInt}
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt128; required = true; help = "Number of criteria")
        ("file"; required = true; help = "Input file")
        ("N"; arg_type = UInt128; required = true; help = "Number of samples")
        (["--seed", "-s"]; arg_type = UInt; help = "Random seed")
    end

    return Args(; parse_args(s; as_symbols = true)...)
end

function main()
    @unpack M, file, N, seed = parse_commandline()

    Random.seed!(seed)

    result = DefaultDict{Vector{Vector{Int}}, Int}(0)

    @unpack labels, nb_paths = GraphFile(load(file))

    K = nb_paths[1]
    @info "Nb paths : $K"

    for i ∈ 1:N
        @debug i
        we = generate_weak_order_ext(labels, nb_paths)
        result[we] += 1
    end

    T = (K / N) * sum(values(result) .^ 2) - N

    println("Uniform : ", T < quantile(Chisq(K - 1), 0.95))
    println("P-value : ", ccdf(Chisq(K - 1), T))
    return 0
end

main()

# Base.ARGS = ["3", "src/julia/WE/3.jld2", "10000"]
# @time main()
# @profview main()
# @code_warntype main()