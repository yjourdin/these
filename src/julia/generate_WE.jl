include("GenerateWeakOrderExt.jl")

using ArgParse


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
        parsed_args["output"]::Union{Nothing, String},
        parsed_args["logging"]::Union{Nothing, IO, String}
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
        open(normpath(output, "nb_paths.bin"), "w+") do nb_paths_io
            open(normpath(output, "edge_labels.bin"), "w+") do edge_labels_io
                generate_WE(nb_paths_io, edge_labels_io, BooleanLattice(Int(M)))

            end
        end
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