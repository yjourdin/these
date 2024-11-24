include("GenerateWeakOrderExt.jl")

using ArgParse
using Distributions
using JLD2

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "M"
            arg_type = UInt128
            required = true
            help = "Number of criteria"
        "dir"
            required = true
            help = "Input directory"
        "N"
            arg_type = UInt128
            required = true
            help = "Number of samples"
        "--seed", "-s"
            arg_type = UInt
            help = "Random seed"
    end

    parsed_args = parse_args(s)

    return (
        parsed_args["M"]::UInt128,
        parsed_args["dir"]::String,
        parsed_args["N"]::UInt128,
        parsed_args["seed"]::Union{UInt, Nothing}
    )
end

function main()
    (M, dir, N, seed) = parse_commandline()

    Random.seed!(seed)

    result = Dict{Vector{Vector{String}}, Int}()

    subsets = elements(BooleanLattice(Int(M)))

    let K::UInt128
        labels = load_object(normpath(dir, "labels.bin"))
        nb_paths = load_object(normpath(dir, "nb_paths.bin"))

        K = nb_paths[1]
        @info K

        for i ∈ 1:N
            @debug i
            we = generate_weak_order_ext(labels, nb_paths, subsets)

            result[we] = get!(result, we, 0) + 1
        end
        
        T = (K / N) * sum(values(result) .^ 2) - N

        println("Uniform : ", T < quantile(Chisq(K - 1), 0.95))
        println("P-value : ", ccdf(Chisq(K - 1), T))
    end
end

main()

# Base.ARGS = ["3", "src/julia/WE/3", "10000"]
# @time main()
# @profview main()