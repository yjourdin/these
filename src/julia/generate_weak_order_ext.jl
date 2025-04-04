include("GenerateWeakOrderExt.jl")

using ArgParse
using JLD2
using UnPack

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt; required = true; help = "Number of criteria")
        ("dir"; required = true; help = "Input directory")
        (["--seed", "-s"]; arg_type = UInt; help = "Random seed")
    end

    return parse_args(s)
end

function main()
    @unpack M, dir, seed = parse_commandline()

    Random.seed!(seed)

    labels = load_object(normpath(dir, "labels.bin"))
    nb_paths = load_object(normpath(dir, "nb_paths.bin"))

    println(generate_weak_order_ext(labels, nb_paths, elements(BooleanLattice(Int(M)))))
    return 0
end

main()

# Base.ARGS = ["5", "src/julia/WE/5"]
# @time main()
# @profview main()
# @code_warntype main()