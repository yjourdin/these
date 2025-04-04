include("GenerateWeakOrderExt.jl")

using ArgParse
using JLD2
using Logging
using UnPack

function fmt(level, _module, group, id, file, line)
    @nospecialize
    color = Logging.default_logcolor(level)
    prefix = string(level == Warn ? "Warning" : string(level), ':')
    suffix::String = ""
    return color, prefix, suffix
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        ("M"; arg_type = UInt; required = true; help = "Number of criteria")
        ("output"; required = true; help = "Output file")
        (["--logging", "-l"]; nargs = '?'; constant = stdout; help = "Logging file")
    end

    return parse_args(s)
end

function main()
    @unpack M, output, logging = parse_commandline()

    if logging isa String
        logging_io = open(logging, "w+")
    else
        logging_io = logging
    end

    if !isnothing(logging_io)
        logger = ConsoleLogger(logging_io, Debug; meta_formatter = fmt)
        global_logger(logger)
    end

    labels, nb_paths = generate_WE(BooleanLattice(Int(M)))

    jldsave(output; labels=labels, nb_paths=nb_paths)

    if logging isa String
        close(logging_io)
    end
    return 0
end

main()

# Base.ARGS = ["4"]
# @time main()
# @profview main()
# @code_warntype main()