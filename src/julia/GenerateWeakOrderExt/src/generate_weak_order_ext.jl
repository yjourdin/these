function generate_weak_order_ext(WE, rng = Random.default_rng())
    @unpack labels, nb_paths = WE
    result = Bitset[]
    N      = length(labels)
    i      = 1
    while i ≠ N
        u  = labels[i]
        Ni = [j for j ∈ (i+1):N if Bit.issubset(u, labels[j])]
        @views i = sample(rng, Ni, FrequencyWeights(nb_paths[Ni], nb_paths[i]))
        push!(result, Bit.setdiff(labels[i], u))
    end
    return result
end