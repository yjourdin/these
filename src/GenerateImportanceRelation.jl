
module GenerateImportanceRelation
import FromFile: @from
using Combinatorics
using Random
@from "GenerateWeakOrder.jl" using GenerateWeakOrder

export generate_importance_relation

function labels_from_int(n)
    return collect(powerset(1:n))
end

function monotonic_index(labels)
    result = empty(labels)

    for label in labels
        index = Vector{Int}()
        for i in findall(l -> length(l) == (length(label) - 1), labels)
            if labels[i] ⊆ label
                push!(index, i)
            end
        end
        push!(result, index)
    end

    return result
end

function check_monotonic(ranking, index, monotonic_vector)
    @inbounds index_to_check = monotonic_vector[index]

    r = @inbounds ranking[index]
    for i in index_to_check
        if @inbounds ranking[i] < r
            return false
        end
    end

    return true
end

function generate_importance_relation(n, rng=Random.default_rng())
    labels = labels_from_int(n)
    ranking = ones(Int, length(labels))
    m = length(labels)
    S = generate_partial_sum(m)

    monotonic_vector = monotonic_index(labels)
    check(ranking, index) = check_monotonic(ranking, index, monotonic_vector)
    
    while !random_ranking_from_partial_sum!(ranking, S, rng, check)
    end

    return ranking, labels
end

end