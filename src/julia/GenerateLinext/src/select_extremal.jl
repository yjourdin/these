function select_M(P, ul, I, h, k, rng = Random.default_rng())
    card_I   = length(I)
    card_III = @chain ul begin
        first
        just_below(P)
        (above(x, P) for x ∈ _)
        (collect(x) for x ∈ _)
        (length(x) == 1 for x ∈ _)
        count
    end
    card_II  = card_I + card_III
    pu, pl   = proba_Th(h, k, card_I, card_II, card_III)
    return sample(rng, [ul; I], ProbabilityWeights([fill(pu, h); fill(pl, card_I)], 1))
end

function select_m(P, ll, I, h, k, rng = Random.default_rng())
    card_I   = length(I)
    card_III = @chain ll begin
        first
        just_above(P)
        (below(x, P) for x ∈ _)
        (collect(x) for x ∈ _)
        (length(x) == 1 for x ∈ _)
        count
    end
    card_II  = card_I + card_III
    pl, pu   = proba_Bh(h, k, card_I, card_II, card_III)
    return sample(rng, [ll; I], ProbabilityWeights([fill(pl, k); fill(pu, card_I)], 1))
end
