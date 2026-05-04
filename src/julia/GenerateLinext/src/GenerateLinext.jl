module GenerateLinext

using ArgMacros
using Bit
using Chain
using Poset:
    above,
    below,
    just_above,
    just_below,
    nv,
    maximals,
    minimals,
    remove_vertex!,
    subset_lattice
using Random
using StatsBase
using UnPack

include("isolated.jl")
include("cardinality.jl")
include("probabilities.jl")
include("select_extremal.jl")
include("generate_linext.jl")

function @main(args)
    @inlinearguments begin
        @argumentoptional UInt seed "-s" "--seed"
        @arghelp "Random seed"

        @positionalrequired UInt M "Number of criteria"
    end

    Random.seed!(seed)

    M |> subset_lattice |> generate_linext! |> println

    return 0
end

end # module GenerateLinext
