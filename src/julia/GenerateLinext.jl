using Random
using Posets
using StatsBase

# Poset

function remove_vertex!(P, labels, v)
    rem_vertex!(P, v)

    labels[v], labels[end] = labels[end], labels[v]
    return pop!(labels)
end

# Isolated

is_isolated_top(P, x)    = isempty(above(P, x))
is_isolated_bottom(P, x) = isempty(below(P, x))
isolated_top(P, A)       = [x for x ∈ A if is_isolated_top(P, x)]
isolated_bottom(P, A)    = [x for x ∈ A if is_isolated_bottom(P, x)]

# Cardinality

cardinality(i)        = count_ones(i)
top_cardinality(A)    = maximum(cardinality, A)
bottom_cardinality(A) = minimum(cardinality, A)

# Sub layers

layer(A, card) = findall(@. cardinality(A) == card)

# Probabilities

function proba_upper_Th(h, k, I, II, III)
    return (1 / h) * (prod(range(h - 1 + k - II + 1; length = II); init = big(1))) / (
        prod(range(h - 1 + k - II + 1; length = II); init = big(1)) +
        I *
        prod(range(h - 1 + k - II + 1; length = III); init = big(1)) *
        prod(range(h + k - I + 1; length = I - 1); init = big(1))
    )
end

function proba_lower_Th(h, k, I, II, III)
    return (
        prod(range(h - 1 + k - II + 1; length = III); init = big(1)) *
        prod(range(h + k - I + 1; length = I - 1); init = big(1))
    ) / (
        prod(range(h - 1 + k - II + 1; length = II); init = big(1)) +
        I *
        prod(range(h - 1 + k - II + 1; length = III); init = big(1)) *
        prod(range(h + k - I + 1; length = I - 1); init = big(1))
    )
end

function proba_upper_Bh(h, k, I, II, III)
    return (
        prod(range(h - II + k - 1 + 1; length = III); init = big(1)) *
        prod(range(h - I + k + 1; length = I - 1); init = big(1))
    ) / (
        prod(range(h - 1 + k - II + 1; length = II); init = big(1)) +
        I *
        prod(range(h - II + k - 1 + 1; length = III); init = big(1)) *
        prod(range(h - I + k + 1; length = I - 1); init = big(1))
    )
end

function proba_lower_Bh(h, k, I, II, III)
    return (1 / k) * (prod(range(h - II + k - 1 + 1; length = II); init = big(1))) / (
        prod(range(h - II + k - 1 + 1; length = II); init = big(1)) +
        I *
        prod(range(h - II + k - 1 + 1; length = III); init = big(1)) *
        prod(range(h - I + k + 1; length = I - 1); init = big(1))
    )
end

function proba_Th(h, k, I, II, III)
    eu    = prod(range(h - 1 + k - II + 1; length = II); init = big(1))
    el    = prod(range(h - 1 + k - II + 1; length = III); init = big(1)) * prod(range(h + k - I + 1; length = I - 1); init = big(1))
    denom = eu + I * el
    pu    = eu / (h * denom)
    pl    = el / denom
    return pu, pl
end

function proba_Bh(h, k, I, II, III)
    el    = prod(range(h - 1 + k - II + 1; length = II); init = big(1))
    eu    = prod(range(h - 1 + k - II + 1; length = III); init = big(1)) * prod(range(h + k - I + 1; length = I - 1); init = big(1))
    denom = eu + I * el
    pl    = el / (k * denom)
    pu    = eu / denom
    return pl, pu
end

# Select extremal

function select_M(P, ul, I, h, k, rng = Random.default_rng())
    card_I   = length(I)
    card_III = ul |> first |> Base.Fix{1}(just_below, P) .|> Base.Fix{1}(above, P) .|> collect .|> length .|> ==(1) |> count
    card_II  = card_I + card_III
    pu, pl   = proba_Th(h, k, card_I, card_II, card_III)
    return sample(rng, [ul; I], ProbabilityWeights([fill(pu, h); fill(pl, card_I)], 1))
end

function select_m(P, ll, I, h, k, rng = Random.default_rng())
    card_I   = length(I)
    card_III = ll |> first |> Base.Fix{1}(just_above, P) .|> Base.Fix{1}(below, P) .|> collect .|> length .|> ==(1) |> count
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
        max = maximals(P)
        M, _ = iterate(max)
        if ~isempty(max)
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

        min = minimals(P)
        m, _ = iterate(min)
        if ~isempty(min)
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
            max = maximals(P)
            M, _ = iterate(max)
            if ~isempty(max)
                I = isolated_top(P, ll)
                M = select_M(P, ul, I, h, k, rng)
            end
            label = labels[M]
            pushfirst!(lmax, label)
            remove_vertex!(P, labels, M)
            (cardinality(label) != top_card) || (top_card = top_cardinality(labels))
        else
            min = minimals(P)
            m, _ = iterate(min)
            if ~isempty(min)
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