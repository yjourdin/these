module GenerateWeakOrderExt

using ArgMacros
using Bit
using Chain
using JLD2
using Poset: subset_lattice, subset_decode
using Random
using StatsBase
using UnPack

include("bit_poset.jl")
include("all_weak_3.jl")
include("WE.jl")
include("generate_WE.jl")
include("generate_weak_order_ext.jl")
include("decode_weak_order_ext.jl")

const DEDEKIND = [3, 6, 20, 168, 7581, 7828354] # OEIS : A000372

function @main(args)
    @inlinearguments begin
        @argumentoptional UInt seed "-s" "--seed"
        @arghelp "Random seed"

        @positionalrequired UInt M "Number of criteria"
    end

    Random.seed!(seed)

    G::WE = let filename = joinpath(dirname(@__DIR__), "WE", "$M.jld2")
        jldopen(filename, "a") do file
            if isempty(file)
                @unpack labels, nb_paths = generate_WE(M)
                @pack! file = labels, nb_paths
            end

            WE(file)
        end
    end

    G |> generate_weak_order_ext |> decode_weak_order_ext |> println

    return 0
end

include("CheckUniformity.jl")

end # module GenerateWeakOrderExt
