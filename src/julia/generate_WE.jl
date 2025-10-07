include("GenerateWeakOrderExt.jl")

using ArgParse
using JLD2
using UnPack

function fmt(level, _module, group, id, file, line)
    @nospecialize
    color          = Logging.default_logcolor(level)
    prefix         = string(level == Warn ? "Warning" : string(level), ':')
    suffix::String = ""
    return color, prefix, suffix
end

@kwdef struct Args
    M       :: UInt
    output  :: String
    logging :: Union{Nothing, Base.TTY, String}
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt; required = true; help = "Number of criteria")
        ("output"; required = true; help = "Output file")
        (["--logging", "-l"]; nargs = '?'; constant = stdout; help = "Logging file")
    end

    return Args(; parse_args(s; as_symbols = true)...)
end

function main()
    @unpack M, output, logging = parse_commandline()

    logging isa String ? logging_io = open(logging, "w+") : logging_io = logging

    isnothing(logging_io) ||
        global_logger(ConsoleLogger(logging_io, Debug; meta_formatter = fmt))

    labels, nb_paths = M |> Posets.subset_lattice |> generate_WE

    jldsave(output; labels, nb_paths)

    logging isa String && close(logging_io)
    return 0
end

main()

# Base.ARGS = ["5", "src/julia/WE/5.jld2"]
# @time main()
# @profview main()
# @code_warntype main()