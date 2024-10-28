module GenerateWeakOrder
using Random
using Memoization

export random_ranking, generate_partial_sum, random_ranking_from_partial_sum!

@memoize function w(m, k)
    if k > m
        return big(0)
    elseif k == 1
        return big(1)
    end
    return k * (w(m - 1, k) + w(m - 1, k - 1))
end

function W(m)
    return sum(k -> w(m, k), 1:m)
end

function generate_partial_sum(m, delta=0.01)
    Wm = W(m)
    k = 0
    S = zeros(1)

    while (Wm - last(S) > delta) && ((length(S) < 2) || S[end] > S[end-1])
        k += 1
        push!(S, last(S) + (big(k)^m) / (big(2)^(k + 1)))
    end

    S[end] = Wm

    return S
end

function random_nb_blocks(S, rng=Random.default_rng())
    Wm = last(S)
    Y = Wm * rand(rng)

    return searchsortedfirst(S, Y) - 1
end

function rand_check!(ranking, k, rng=Random.default_rng(), check=(ranking, i) -> true)
    for i in eachindex(ranking)
        @inbounds ranking[i] = rand(rng, 1:k)
        if ! check(ranking, i)
            return false
        end
    end

    return true
end

function random_ranking_from_blocks!(ranking, k, rng=Random.default_rng(), check=(ranking, i) -> true)
    # return rand!(rng, ranking, 1:k)
    return rand_check!(ranking, k, rng, check)
end

function random_ranking_from_partial_sum!(ranking, S, rng=Random.default_rng(), check=(ranking, i) -> true)
    K = random_nb_blocks(S, rng)

    return random_ranking_from_blocks!(ranking, K, rng, check)
end

function random_ranking(m, rng=Random.default_rng(), delta=0.01)
    ranking = ones(Int, m)
    S = generate_partial_sum(m, delta)

    random_ranking_from_partial_sum!(ranking, S, rng)

    return ranking
end

end