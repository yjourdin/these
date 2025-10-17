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