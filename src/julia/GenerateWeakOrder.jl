using Memoization
using Random
using StatsBase

@memoize function w(m, k)
    k > m && return big(0)
    k == 1 && return big(1)
    return k * (w(m - 1, k) + w(m - 1, k - 1))
end

W(m) = sum(w(m, k) for k ∈ 1:m)

function generate_partial_sum(m, delta = 0.01)
    Wm = W(m)
    k  = 0
    S  = BigFloat[0]

    while (Wm - last(S) > delta) && ((length(S) < 2) || S[end] > S[end - 1])
        k += 1
        push!(S, last(S) + (big(k)^m) / (2^(k + 1)))
    end

    S[end] = Wm

    return S
end

function random_nb_blocks(S, rng = Random.default_rng())
    Wm = last(S)
    Y  = Wm * rand(rng)

    return searchsortedlast(S, Y)
end

function random_ranking_from_blocks(m, k, rng = Random.default_rng())
    return denserank(rand(rng, 1:k, m))
end

function random_ranking(m, S, rng = Random.default_rng())
    K = random_nb_blocks(S, rng)

    return random_ranking_from_blocks(m, K, rng)
end