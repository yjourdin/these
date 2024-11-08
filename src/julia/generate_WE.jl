include("GenerateWeakOrderExt.jl")

using ArgParse


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "m"
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

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    if !isdir(parsed_args["output"])
        mkpath(parsed_args["output"])
        open(normpath(parsed_args["output"], "nb_paths.bin"), "w+") do nb_paths_io
            open(normpath(parsed_args["output"], "edge_labels.bin"), "w+") do edge_labels_io
                if parsed_args["logging"] isa String
                    logging_io = open(parsed_args["logging"], "w+")
                else
                    logging_io = parsed_args["logging"]
                end

                generate_WE(nb_paths_io, edge_labels_io, BooleanLattice(Int(parsed_args["m"])), logging_io)

                if parsed_args["logging"] isa String
                    close(logging_io)
                end
            end
        end
    end
end

main()

# ARGS=["5"]
# @time main()
# @profview main()