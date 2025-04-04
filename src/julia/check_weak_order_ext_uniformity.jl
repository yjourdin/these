include("GenerateWeakOrderExt.jl")

using ArgParse
using Distributions
using JLD2
using UnPack

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt128; required = true; help = "Number of criteria")
        ("input"; required = true; help = "Input directory")
        ("N"; arg_type = UInt128; required = true; help = "Number of samples")
        (["--seed", "-s"]; arg_type = UInt; help = "Random seed")
    end

    return parse_args(s)
end

function main()
    @unpack M, input, N, seed = parse_commandline()

    Random.seed!(seed)

    result = Dict{Vector{Vector{String}}, Int}()

    subsets = elements(BooleanLattice(Int(M)))

    @unpack labels, nb_paths = jldopen(input, "r")

    K = nb_paths[1]
    @info "Nb paths : $K"

    for i ∈ 1:N
        @debug i
        we = generate_weak_order_ext(labels, nb_paths, subsets)

        result[we] = get!(result, we, 0) + 1
    end

    T = (K / N) * sum(values(result) .^ 2) - N

    println("Uniform : ", T < quantile(Chisq(K - 1), 0.95))
    println("P-value : ", ccdf(Chisq(K - 1), T))
    return 0
end

main()

# Base.ARGS = ["3", "src/julia/WE/3", "10000"]
# @time main()
# @profview main()
# @code_warntype main()