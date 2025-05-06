include("GenerateWeakOrder.jl")

using ArgParse
using JLD2
using UnPack

@kwdef struct Args
    M      :: UInt
    output :: Union{Nothing, String}
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt; required = true; help = "Number of criteria")
        (["--output", "-o"]; help = "Output file")
    end

    return Args(; parse_args(s; as_symbols = true)...)
end

function main()
    @unpack M, output = parse_commandline()

    S = generate_partial_sum(M)

    !isnothing(output) ? save_object(output, S) : println(S)
    return 0
end

main()

# Base.ARGS = ["10"]
# @time main()
# @profview main()
# @code_warntype main()