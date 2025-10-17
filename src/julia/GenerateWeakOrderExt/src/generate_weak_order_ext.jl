function generate_weak_order_ext(WE, rng = Random.default_rng())
    @unpack labels, nb_paths = WE
    result = Vector{Vector{Int}}[]
    N      = length(labels)
    i      = 1
    while i ≠ N
        Ni       = collect(successors(labels, i))
        @views j = sample(rng, Ni, FrequencyWeights(nb_paths[Ni], nb_paths[i]))
        @chain labels begin
            Bit.setdiff(_[j], _[i])
            decode
            collect
            @. subset_decode
            @. collect
            push!(result, _)
        end
        i = j
    end
    return result
end