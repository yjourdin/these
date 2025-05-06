using Random
using Posets
using StatsBase

# get_only

function get_only(it)
    local x
    try
        x = only(it)
    catch e
        e isa ArgumentError ? (return nothing) : rethrow(e)
    else
        return x
    end
end

# Poset

function remove_vertex!(P, labels, v)
    rem_vertex!(P, v)

    labels[v], labels[end] = labels[end], labels[v]
    return pop!(labels)
end

# Isolated

is_isolated_top(P, x)    = isempty(above(P, x))
is_isolated_bottom(P, x) = isempty(below(P, x))
isolated_top(P, A)       = filter(x -> is_isolated_top(P, x), A)
isolated_bottom(P, A)    = filter(x -> is_isolated_bottom(P, x), A)

# Cardinality

cardinality(i)        = count_ones(i)
top_cardinality(A)    = maximum(cardinality.(A))
bottom_cardinality(A) = minimum(cardinality.(A))

# Sub layers

layer(A, card) = findall(cardinality.(A) .== card)

# Probabilities

function proba_upper_Th(h, k, I, II, III)
    return (1 / h) * (prod(big(h - 1 + k - II + i) for i ∈ 1:II; init = big(1))) / (
        prod(big(h - 1 + k - II + i) for i ∈ 1:II; init = big(1)) +
        I *
        prod(big(h - 1 + k - II + i) for i ∈ 1:III; init = big(1)) *
        prod(big(h + k - I + i) for i ∈ 1:(I - 1); init = big(1))
    )
end

function proba_lower_Th(h, k, I, II, III)
    return (
        prod(big(h - 1 + k - II + i) for i ∈ 1:III; init = big(1)) *
        prod(big(h + k - I + i) for i ∈ 1:(I - 1); init = big(1))
    ) / (
        prod(big(h - 1 + k - II + i) for i ∈ 1:II; init = big(1)) +
        I *
        prod(big(h - 1 + k - II + i) for i ∈ 1:III; init = big(1)) *
        prod(big(h + k - I + i) for i ∈ 1:(I - 1); init = big(1))
    )
end

function proba_upper_Bh(h, k, I, II, III)
    return (
        prod(big(h - II + k - 1 + i) for i ∈ 1:III; init = big(1)) *
        prod(big(h - I + k + i) for i ∈ 1:(I - 1); init = big(1))
    ) / (
        prod(big(h - II + k - 1 + i) for i ∈ 1:II; init = big(1)) +
        I *
        prod(big(h - II + k - 1 + i) for i ∈ 1:III; init = big(1)) *
        prod(big(h - I + k + i) for i ∈ 1:(I - 1); init = big(1))
    )
end

function proba_lower_Bh(h, k, I, II, III)
    return (1 / k) * (prod(big(h - II + k - 1 + i) for i ∈ 1:II; init = big(1))) / (
        prod(big(h - II + k - 1 + i) for i ∈ 1:II; init = big(1)) +
        I *
        prod(big(h - II + k - 1 + i) for i ∈ 1:III; init = big(1)) *
        prod(big(h - I + k + i) for i ∈ 1:(I - 1); init = big(1))
    )
end

function proba_Th(h, k, I, II, III)
    eu    = prod(big(h - 1 + k - II + i) for i ∈ 1:II; init = big(1))
    el    = prod(big(h - 1 + k - II + i) for i ∈ 1:III; init = big(1)) * prod(big(h + k - I + i) for i ∈ 1:(I - 1); init = big(1))
    denom = eu + I * el
    pu    = eu / (h * denom)
    pl    = el / denom
    return pu, pl
end

function proba_Bh(h, k, I, II, III)
    el    = prod(big(h - 1 + k - II + i) for i ∈ 1:II; init = big(1))
    eu    = prod(big(h - 1 + k - II + i) for i ∈ 1:III; init = big(1)) * prod(big(h + k - I + i) for i ∈ 1:(I - 1); init = big(1))
    denom = eu + I * el
    pl    = el / (k * denom)
    pu    = eu / denom
    return pl, pu
end

# Select extremal

function select_M(P, ul, I, h, k, rng = Random.default_rng())
    card_I   = length(I)
    card_III = count(x -> isempty(Iterators.drop(above(P, x), 1)), just_below(P, first(ul)))
    card_II  = card_I + card_III
    pu, pl   = proba_Th(h, k, card_I, card_II, card_III)
    return sample(rng, [ul; I], ProbabilityWeights([fill(pu, h); fill(pl, card_I)], 1))
end

function select_m(P, ll, I, h, k, rng = Random.default_rng())
    card_I   = length(I)
    card_III = count(x -> isempty(Iterators.drop(below(P, x), 1)), just_above(P, first(ll)))
    card_II  = card_I + card_III
    pl, pu   = proba_Bh(h, k, card_I, card_II, card_III)
    return sample(rng, [ll; I], ProbabilityWeights([fill(pl, k); fill(pu, card_I)], 1))
end

# generate_linext

function generate_linext!(P, rng = Random.default_rng())
    lmin        = Int[]
    lmax        = Int[]
    labels      = collect(0:(nv(P) - 1))
    top_card    = top_cardinality(labels)
    bottom_card = bottom_cardinality(labels)

    while (length(labels) >= 2) && (top_card - bottom_card > 1)
        M = get_only(maximals(P))
        if isnothing(M)
            ul = layer(labels, top_card)
            ll = layer(labels, top_card - 1)
            h  = length(ul)
            k  = length(ll)
            I  = isolated_top(P, ll)
            M  = select_M(P, ul, I, h, k, rng)
        end
        label = labels[M]
        pushfirst!(lmax, label)
        remove_vertex!(P, labels, M)
        (cardinality(label) != top_card) || (top_card = top_cardinality(labels))

        m = get_only(minimals(P))
        if isnothing(m)
            ul = layer(labels, bottom_card + 1)
            ll = layer(labels, bottom_card)
            h  = length(ul)
            k  = length(ll)
            I  = isolated_bottom(P, ul)
            m  = select_m(P, ll, I, h, k, rng)
        end
        label = labels[m]
        push!(lmin, label)
        remove_vertex!(P, labels, m)
        (cardinality(label) != bottom_card) ||
            (bottom_card = bottom_cardinality(labels))
    end

    while (length(labels) >= 2) && (top_card - bottom_card == 1)
        ul = layer(labels, top_card)
        ll = layer(labels, bottom_card)
        h  = length(ul)
        k  = length(ll)
        if h <= k
            M = get_only(maximals(P))
            if isnothing(M)
                I = isolated_top(P, ll)
                M = select_M(P, ul, I, h, k, rng)
            end
            label = labels[M]
            pushfirst!(lmax, label)
            remove_vertex!(P, labels, M)
            (cardinality(label) != top_card) || (top_card = top_cardinality(labels))
        else
            m = get_only(minimals(P))
            if isnothing(m)
                I = isolated_bottom(P, ul)
                m = select_m(P, ll, I, h, k, rng)
            end
            label = labels[m]
            push!(lmin, label)
            remove_vertex!(P, labels, m)
            (cardinality(label) != bottom_card) ||
                (bottom_card = bottom_cardinality(labels))
        end
    end

    append!(lmin, shuffle!(labels))

    return [lmin; lmax]
end