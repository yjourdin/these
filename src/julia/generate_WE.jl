include("GenerateWeakOrderExt.jl")

using ArgParse
using JLD2


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "M"
            arg_type = UInt
            required = true
            help = "Number of criteria"
        "--output", "-o"
            default = pwd()
            help = "Output directory"
        "--logging", "-l"
            nargs = '?'
            constant = stdout
            help = "Logging file"
    end

    parsed_args = parse_args(s)

    return (
        parsed_args["M"]::UInt,
        parsed_args["output"]::Union{Nothing,String},
        parsed_args["logging"]::Union{Nothing,IO,String}
    )
end

function main()
    (M, output, logging) = parse_commandline()

    if logging isa String
        logging_io = open(logging, "w+")
    else
        logging_io = logging
    end

    if !isnothing(logging_io)
        logger = SimpleLogger(logging_io, Debug)
        global_logger(logger)
    end

    if !isdir(output)
        mkpath(output)
        labels, nb_paths = generate_WE(BooleanLattice(Int(M)))

        save_object(normpath(output, "labels.bin"), labels)
        save_object(normpath(output, "nb_paths.bin"), nb_paths)
    else
        @warn "directory already exist"
    end

    if logging isa String
        close(logging_io)
    end
end

main()

# Base.ARGS = ["5"]
# @time main()
# @profview main()