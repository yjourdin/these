module CheckUniformity

using ArgMacros
using Bit
using Chain
using DataStructures
using Distributions
using JLD2
using Poset: subset_decode
using Random
using StatsAPI: pvalue
using StatsBase
using UnPack

include("WE.jl")
include("generate_weak_order_ext.jl")

function @main(args)
    @inlinearguments begin
        @argumentoptional UInt seed "-s" "--seed"
        @arghelp "Random seed"

        @positionalrequired UInt M "Number of criteria"

        @positionaloptional UInt128 N_opt "Number of samples"
    end

    Random.seed!(seed)

    G::WE = let filename = joinpath(dirname(@__DIR__), "WE", "$M.jld2")
        jldopen(filename, "r") do file
            return WE(file)
        end
    end

    K = G.nb_paths[begin]
    N_min = 5 * K
    @info "Minimum expected samples : $N_min"
    N = isnothing(N_opt) ? N_min : N_opt

    c = counter(generate_weak_order_ext(G) for _ ∈ 1:N)

    T = (K / N) * sum(x^2 for x ∈ values(c)) - N

    dist = Chisq(K - 1)

    p = pvalue(dist, T; tail = :right)

    println("Uniform : ", p > 0.05)
    println("P-value : ", p)

    return 0
end

end # module CheckUniformity