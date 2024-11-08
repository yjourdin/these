include("GenerateWeakOrderExt.jl")

using ArgParse


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "M"
            arg_type = UInt
            required = true
            help = "Number of criteria"
        "dir"
            required = true
            help = "Input directory"
        "--seed", "-s"
            arg_type = UInt
            help = "Random seed"
    end

    parsed_args = parse_args(s)

    return (
        parsed_args["M"]::UInt,
        parsed_args["dir"]::String,
        parsed_args["seed"]::Union{UInt, Nothing}
    )
end

function main()
    (M, dir, seed) = parse_commandline()

    Random.seed!(seed)

    open(normpath(dir, "nb_paths.bin"), "r") do nb_paths_io
        open(normpath(dir, "edge_labels.bin"), "r") do edge_labels_io
            nb_paths = mmap(nb_paths_io, Vector{Int128})
            edge_labels = mmap(edge_labels_io, Vector{Int128})
            
            println(generate_weak_order_ext(nb_paths, edge_labels, elements(BooleanLattice(Int(M)))))
        end
    end
end

main()

# Base.ARGS = ["5", "src/julia/WE/5"]
# @time main()
# @profview main()