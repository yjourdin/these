using Random
using SimplePosets
using StatsBase

# Isolated

function is_isolated_top(P, x)
    return isempty(above(P, x))
end

function is_isolated_bottom(P, x)
    return isempty(below(P, x))
end

function isolated_top(P, A)
    return filter(x -> is_isolated_top(P, x), A)
end

function isolated_bottom(P, A)
    return filter(x -> is_isolated_bottom(P, x), A)
end

# Cardinality
function cardinality(str)
    return count_ones(parse(UInt, str, base=2))
end

function top_cardinality(P)
    return maximum(cardinality.(maximals(P)))
end

function bottom_cardinality(P)
    return minimum(cardinality.(minimals(P)))
end

# Sub layers

function layer(elements, card)
    return filter(x -> cardinality(x) == card, elements)
end

# Probabilities

function proba_upper_Th(h, k, I, II, III)
    return (1 / h) * (prod(big(h - 1 + k - II + i) for i in 1:II; init=big(1))) / (prod(big(h - 1 + k - II + i) for i in 1:II; init=big(1)) + I * prod(big(h - 1 + k - II + i) for i in 1:III; init=big(1)) * prod(big(h + k - I + i) for i in 1:(I-1); init=big(1)))
end

function proba_lower_Th(h, k, I, II, III)
    return (prod([big(h - 1 + k - II + i) for i in 1:III]) * prod([big(h + k - I + i) for i in 1:(I-1)])) / (prod([big(h - 1 + k - II + i) for i in 1:II]) + I * prod([big(h - 1 + k - II + i) for i in 1:III]) * prod([big(h + k - I + i) for i in 1:(I-1)]))
end

function proba_upper_Bh(hh, k, I, II, III)
    return (prod([big(h - II + k - 1 + i) for i in 1:III]) * prod([big(h - I + k + i) for i in 1:(I-1)])) / (prod([big(h - II + k - 1 + i) for i in 1:II]) + I * prod([big(h - II + k - 1 + i) for i in 1:III]) * prod([big(h - I + k + i) for i in 1:(I-1)]))
end

function proba_lower_Bh(h, k, I, II, III)
    return (1 / k) * (prod([big(h - II + k - 1 + i) for i in 1:II])) / (prod([big(h - II + k - 1 + i) for i in 1:II]) + I * prod([big(h - II + k - 1 + i) for i in 1:III]) * prod([big(h - I + k + i) for i in 1:(I-1)]))
end

function proba_Th(h, k, I, II, III)
    A = [big(h - 1 + k - II + i) for i in 1:II]
    eu = prod(A)
    el = prod(view(A,1:III)) * prod([big(h + k - I + i) for i in 1:(I-1)])
    # eu = prod(big(h - 1 + k - II + i) for i in 1:II; init=big(1))
    # el = prod(big(h - 1 + k - II + i) for i in 1:III; init=big(1)) * prod(big(h + k - I + i) for i in 1:(I-1); init=big(1))
    denom = eu + I * el
    pu = eu / (h * denom)
    pl = el / denom
    return pu, pl
end

function proba_Bh(h, k, I, II, III)
    A = [big(h - 1 + k - II + i) for i in 1:II]
    el = prod(A)
    eu = prod(view(A,1:III)) * prod([big(h + k - I + i) for i in 1:(I-1)])
    # el = prod(big(h - 1 + k - II + i) for i in 1:II; init=big(1))
    # eu = prod(big(h - 1 + k - II + i) for i in 1:III; init=big(1)) * prod(big(h + k - I + i) for i in 1:(I-1); init=big(1))
    denom = eu + I * el
    pl = el / (k * denom)
    pu = eu / denom
    return pl, pu
end

# Select extremal

function select_M(P, ul, I, h, k, rng)
    card_I = length(I)
    III = count(x -> length(above(P, x)) == 1, below(P, first(ul)))
    II = card_I + III
    # pu = proba_upper_Th(h, k, card_I, II, III)
    # pl = proba_lower_Th(h, k, card_I, II, III)
    pu, pl = proba_Th(h, k, card_I, II, III)
    return sample(rng, [collect(ul); collect(I)], ProbabilityWeights([fill(pu, h); fill(pl, card_I)]))
end

function select_m(P, ll, I, h, k, rng)
    card_I = length(I)
    III = count(x -> length(below(P, x)) == 1, above(P, first(ll)))
    II = card_I + III
    # pu = proba_upper_Bh(h, k, card_I, II, III)
    # pl = proba_lower_Bh(h, k, card_I, II, III)
    pl, pu = proba_Bh(h, k, card_I, II, III)
    return sample(rng, [collect(ll); collect(I)], ProbabilityWeights([fill(pl, k); fill(pu, card_I)]))
end

# generate_linext

function generate_linext(P, rng)
    H = deepcopy(P)
    lmin = String[]
    lmax = String[]

    elem = Set(elements(H))
    top_card = top_cardinality(H)
    bottom_card = bottom_cardinality(H)

    # Th = TopTwoLayers(H)
    # Bh = BottomTwoLayers(H)

    while top_card - bottom_card > 1
        max = maximals(H)
        if length(max) == 1
            M = max[1]
        else
            ul = layer(elem, top_card)
            ll = layer(elem, top_card - 1)
            h = length(ul)
            k = length(ll)
            I = isolated_top(H, ll)
            M = select_M(H, ul, I, h, k, rng)
        end
        pushfirst!(lmax, M)
        delete!(H, M)
        delete!(elem, M)
        if cardinality(M) == top_card
            top_card = top_cardinality(H)
        end

        min = minimals(H)
        if length(min) == 1
            m = min[1]
        else
            ul = layer(elem, bottom_card + 1)
            ll = layer(elem, bottom_card)
            h = length(ul)
            k = length(ll)
            I = isolated_bottom(H, ul)
            m = select_m(H, ll, I, h, k, rng)
        end
        push!(lmin, m)
        delete!(H, m)
        delete!(elem, m)
        if cardinality(m) == bottom_card
            bottom_card = bottom_cardinality(H)
        end
    end

    while top_card - bottom_card == 1
        ul = layer(elem, top_card)
        ll = layer(elem, bottom_card)
        h = length(ul)
        k = length(ll)
        if h <= k
            max = maximals(H)
            if length(max) == 1
                M = max[1]
            else
                I = isolated_top(H, ll)
                M = select_M(H, ul, I, h, k, rng)
            end
            pushfirst!(lmax, M)
            delete!(H, M)
            delete!(elem, M)
            if cardinality(M) == top_card
                top_card = top_cardinality(H)
            end
        else
            min = minimals(H)
            if length(min) == 1
                m = min[1]
            else
                I = isolated_bottom(H, ul)
                m = select_m(H, ll, I, h, k, rng)
            end
            push!(lmin, m)
            delete!(H, m)
            delete!(elem, m)
            if cardinality(m) == bottom_card
                bottom_card = bottom_cardinality(H)
            end
        end
    end

    while !isempty(elem)
        x = sample(rng, collect(elem))
        push!(lmin, x)
        delete!(elem, x)
    end

    return [lmin; lmax]
end

# Main

# Random.seed!(parse(Int, ARGS[2]))

# println(generate_linext(BooleanLattice(parse(Int, ARGS[1])), Random.default_rng()))

@time generate_linext(BooleanLattice(parse(Int, "15")), Random.default_rng())
