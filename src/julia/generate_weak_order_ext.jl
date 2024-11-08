include("GenerateWeakOrderExt.jl")

using ArgParse


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "m"
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

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    Random.seed!(parsed_args["seed"])

    open(normpath(parsed_args["dir"], "nb_paths.bin"), "w+") do nb_paths_io
        open(normpath(parsed_args["dir"], "edge_labels.bin"), "w+") do edge_labels_io
            println(generate_weak_order_ext(nb_paths_io, edge_labels_io, elements(BooleanLattice(Int(parsed_args["m"])))))
        end
    end
end

main()

# ARGS=["5", "src/julia/WE/5"]
# @time main()
# @profview main()