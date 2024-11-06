using MetaGraphs
using Random
using StatsBase

function generate_weak_order_ext(WE, rng=Random.default_rng())
    result = Vector{String}[]

    u = 1
    Nu = MetaGraphs.outneighbors(WE, u)

    while !isempty(Nu)
        v = sample(rng, Nu, FrequencyWeights([get_prop(WE, v, :nb_paths) for v in Nu], get_prop(WE, u, :nb_paths)))
        push!(result, get_prop(WE, u, v, :label))
        u = v
        Nu = MetaGraphs.outneighbors(WE, u)
    end

    return result
end

