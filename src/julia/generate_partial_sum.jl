include("GenerateWeakOrder.jl")

using ArgParse
using JLD2
using UnPack

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt; required = true; help = "Number of criteria")
        (["--output", "-o"]; help = "Output file")
    end

    return parse_args(s)
end

function main()
    @unpack M, output = parse_commandline()

    S = generate_partial_sum(M)

    if isnothing(output)
        println(S)
    else
        save_object(output, S)
    end
    return 0
end

main()

# Base.ARGS = ["10"]
# @time main()
# @profview main()
# @code_warntype main()