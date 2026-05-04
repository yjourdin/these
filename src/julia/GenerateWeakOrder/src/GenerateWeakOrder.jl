module GenerateWeakOrder

using ArgMacros
using JLD2
using Memoization
using Random
using StatsBase
using UnPack

include("number_weak_orders.jl")
include("generate_partial_sum.jl")
include("random.jl")

function @main(args)
    @inlinearguments begin
        @argumentoptional UInt seed "-s" "--seed"
        @arghelp "Random seed"

        @positionalrequired UInt M "Number of criteria"
    end

    Random.seed!(seed)

    S = let filename = joinpath(dirname(@__DIR__), "S", "$M.jld2")
        if isfile(filename)
            load_object(filename)::Vector{PartialSumType}
        else
            S = generate_partial_sum(M)
            save_object(filename, S)
            S
        end
    end

    println(random_ranking(M, S))

    return 0
end

end # module GenerateWeakOrder
