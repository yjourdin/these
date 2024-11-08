include("GenerateWeakOrderExt.jl")

using ArgParse
using Distributions

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
        open(normpath(dir, "nb_paths.bin"), "r") do nb_paths_io
            open(normpath(dir, "edge_labels.bin"), "r") do edge_labels_io
                nb_paths = mmap(nb_paths_io, Vector{UInt128})
                edge_labels = mmap(edge_labels_io, Vector{UInt128})

                K = nb_paths[1]
                @info K

                for i ∈ 1:N
                    @debug i
                    we = generate_weak_order_ext(nb_paths, edge_labels, subsets)

                    result[we] = get!(result, we, 0) + 1
                end
            end
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