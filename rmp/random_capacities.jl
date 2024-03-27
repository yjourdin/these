using Random
using SimplePosets
using StatsBase
import SimplePosets: delete!

# Isolated methods

function is_isolated_top(P::SimplePoset, x)
    return isempty(above(P, x))
end

function is_isolated_bottom(P::SimplePoset, x)
    return isempty(below(P, x))
end

function isolated_top(P::SimplePoset, A::Vector)
    return filter(x -> is_isolated_top(P, x), A)
end

function isolated_bottom(P::SimplePoset, A::Vector)
    return filter(x -> is_isolated_bottom(P, x), A)
end

# Cardinality methods
function cardinality(str::String)
    return count_ones(parse(UInt, str, base=2))
end

function top_cardinality(P::SimplePoset)
    return maximum(cardinality.(maximals(P)))
end

function bottom_cardinality(P::SimplePoset)
    return minimum(cardinality.(minimals(P)))
end

# Two layers methods

function top_two_layers(P::SimplePoset, top_card::Int)
    layers = filter(x -> cardinality(x) >= top_card - 1, elements(P))
    return induce(P, Set(layers))
end

function bottom_two_layers(P::SimplePoset, bottom_card::Int)
    layers = layers = filter(x -> cardinality(x) <= bottom_card + 1, elements(P))
    return induce(P, Set(layers))
end

# Two layers struct

mutable struct TopTwoLayers{T}
    poset::SimplePoset{T}
    layers::SimplePoset{T}
    top_card::Int
end

mutable struct BottomTwoLayers{T}
    poset::SimplePoset{T}
    layers::SimplePoset{T}
    bottom_card::Int
end

# Constructors

function TopTwoLayers(P::SimplePoset)
    top_card = top_cardinality(P)
    TopTwoLayers(P, top_two_layers(P, top_card), top_card)
end

function BottomTwoLayers(P::SimplePoset)
    bottom_card = bottom_cardinality(P)
    BottomTwoLayers(P, bottom_two_layers(P, bottom_card), bottom_card)
end

# Sub layers

function layer(P::SimplePoset, card::Int)
    return filter(x -> cardinality(x) == card, elements(P))
end

function upper_layer(Th::TopTwoLayers)
    return layer(Th.layers, Th.top_card)
end

function lower_layer(Th::TopTwoLayers)
    return layer(Th.layers, Th.top_card - 1)
end

function upper_layer(Bh::BottomTwoLayers)
    return layer(Bh.layers, Bh.bottom_card + 1)
end

function lower_layer(Bh::BottomTwoLayers)
    return layer(Bh.layers, Bh.bottom_card)
end

# Delete methods

function delete!(Th::TopTwoLayers, x)
    if has(Th.layers, x)
        delete!(Th.layers, x)
        if cardinality(x) == Th.top_card
            Th.top_card = top_cardinality(Th.layers)
            if Th.top_card < cardinality(x)
                Th.layers = top_two_layers(Th.poset, Th.top_card)
            end
        end
    end
end

function delete!(Bh::BottomTwoLayers, x)
    if has(Bh.layers, x)
        delete!(Bh.layers, x)
        if cardinality(x) == Bh.bottom_card
            Bh.bottom_card = bottom_cardinality(Bh.layers)
            if Bh.bottom_card > cardinality(x)
                Bh.layers = bottom_two_layers(Bh.poset, Bh.bottom_card)
            end
        end
    end
end

function proba_upper_Th(h::Int, k::Int, I::Int, II::Int, III::Int)
    return (1 / h) * (prod(big(h - 1 + k - II + i) for i in 1:II; init=big(1))) / (prod(big(h - 1 + k - II + i) for i in 1:II; init=big(1)) + I * prod(big(h - 1 + k - II + i) for i in 1:III; init=big(1)) * prod(big(h + k - I + i) for i in 1:(I-1); init=big(1)))
end

function proba_lower_Th(h::Int, k::Int, I::Int, II::Int, III::Int)
    return (prod([big(h - 1 + k - II + i) for i in 1:III]) * prod([big(h + k - I + i) for i in 1:(I-1)])) / (prod([big(h - 1 + k - II + i) for i in 1:II]) + I * prod([big(h - 1 + k - II + i) for i in 1:III]) * prod([big(h + k - I + i) for i in 1:(I-1)]))
end

function proba_upper_Bh(h::Int, k::Int, I::Int, II::Int, III::Int)
    return (prod([big(h - II + k - 1 + i) for i in 1:III]) * prod([big(h - I + k + i) for i in 1:(I-1)])) / (prod([big(h - II + k - 1 + i) for i in 1:II]) + I * prod([big(h - II + k - 1 + i) for i in 1:III]) * prod([big(h - I + k + i) for i in 1:(I-1)]))
end

function proba_lower_Bh(h::Int, k::Int, I::Int, II::Int, III::Int)
    return (1 / k) * (prod([big(h - II + k - 1 + i) for i in 1:II])) / (prod([big(h - II + k - 1 + i) for i in 1:II]) + I * prod([big(h - II + k - 1 + i) for i in 1:III]) * prod([big(h - I + k + i) for i in 1:(I-1)]))
end

function proba_Th(h::Int, k::Int, I::Int, II::Int, III::Int)
    eu = prod(big(h - 1 + k - II + i) for i in 1:II; init=big(1))
    el = prod(big(h - 1 + k - II + i) for i in 1:III; init=big(1)) * prod(big(h + k - I + i) for i in 1:(I-1); init=big(1))
    pu = eu / (h * (eu + I * el))
    pl = el / (eu + I * el)
    return pu, pl
end

function proba_Bh(h::Int, k::Int, I::Int, II::Int, III::Int)
    el = prod(big(h - 1 + k - II + i) for i in 1:II; init=big(1))
    eu = prod(big(h - 1 + k - II + i) for i in 1:III; init=big(1)) * prod(big(h + k - I + i) for i in 1:(I-1); init=big(1))
    pl = el / (k * (eu + I * el))
    pu = eu / (eu + I * el)
    return pl, pu
end

function select_M(Th::SimplePoset, ul::Vector, h::Int, k::Int, I::Vector, rng::AbstractRNG)
    card_I = length(I)
    III = count(x -> above(Th, x) == [ul[1]], below(Th, ul[1]))
    II = card_I + III
    # pu = proba_upper_Th(h, k, card_I, II, III)
    # pl = proba_lower_Th(h, k, card_I, II, III)
    pu, pl = proba_Th(h, k, card_I, II, III)
    return sample(rng, [ul; I], ProbabilityWeights([fill(pu, length(ul)); fill(pl, card_I)]))
end

function select_m(Bh::SimplePoset, ll::Vector, h::Int, k::Int, I::Vector, rng::AbstractRNG)
    card_I = length(I)
    III = count(x -> below(Bh, x) == [ll[1]], above(Bh, ll[1]))
    II = card_I + III
    # pu = proba_upper_Bh(h, k, card_I, II, III)
    # pl = proba_lower_Bh(h, k, card_I, II, III)
    pl, pu = proba_Bh(h, k, card_I, II, III)
    return sample(rng, [ll; I], ProbabilityWeights([fill(pl, length(ll)); fill(pu, card_I)]))
end

function generate_linext(P::SimplePoset, rng::AbstractRNG)
    H = deepcopy(P)
    lmin = String[]
    lmax = String[]

    Th = TopTwoLayers(H)
    Bh = BottomTwoLayers(H)

    while Th.top_card - Bh.bottom_card > 1
        if length(maximals(Th.layers)) == 1
            M = maximals(H)[1]
        else
            ul = upper_layer(Th)
            ll = lower_layer(Th)
            h = length(ul)
            k = length(ll)
            I = isolated_top(Th.layers, ll)
            M = select_M(Th.layers, ul, h, k, I, rng)
        end
        pushfirst!(lmax, M)
        delete!(H, M)
        delete!(Th, M)
        delete!(Bh, M)

        if length(minimals(Bh.layers)) == 1
            m = minimals(H)[1]
        else
            ul = upper_layer(Bh)
            ll = lower_layer(Bh)
            h = length(ul)
            k = length(ll)
            I = isolated_bottom(Bh.layers, ul)
            m = select_m(Bh.layers, ll, h, k, I, rng)
        end
        push!(lmin, m)
        delete!(H, m)
        delete!(Th, m)
        delete!(Bh, m)
    end

    while height(H) == 2
        ul = maximals(H)
        ll = minimals(H)
        h = length(ul)
        k = length(ll)
        if h <= k
            if h == 1
                M = ul[1]
            else
                I = isolated_top(H, ll)
                M = select_M(H, ul, h, k, I, rng)
            end
            pushfirst!(lmax, M)
            delete!(H, M)
        else
            if k == 1
                m = ll[1]
            else
                I = isolated_bottom(H, ul)
                m = select_m(H, ll, h, k, I, rng)
            end
            push!(lmin, m)
            delete!(H, m)
        end
    end

    while card(H) > 0
        x = sample(rng, elements(H))
        push!(lmin, x)
        delete!(H, x)
    end

    return [lmin; lmax]
end

# Random.seed!(parse(Int, ARGS[2]))

# println(generate_linext(BooleanLattice(parse(Int, ARGS[1])), Random.default_rng()))

@profview generate_linext(BooleanLattice(parse(Int, "12")), Random.default_rng())
