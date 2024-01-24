using Random
using SimplePosets
using StatsBase

function cardinality(str::String)
    return count_ones(parse(UInt, str, base=2))
end

function height(P::SimplePoset)
    cards = cardinality.(elements(P))
    return length(unique(cards))
end

function isolated(P::SimplePoset)
    return filter(x -> above(P, x) == below(P, x), elements(P))
end

function upper_layer(P::SimplePoset)
    top_card = maximum(cardinality.(elements(P)))
    return filter(x -> cardinality(x) == top_card, elements(P))
end

function lower_layer(P::SimplePoset)
    bottom_card = minimum(cardinality.(elements(P)))
    return filter(x -> cardinality(x) == bottom_card, elements(P))
end

function top_layers(P::SimplePoset)
    top_card = maximum(cardinality.(elements(P)))
    layers = filter(x -> cardinality(x) >= top_card - 1, elements(P))
    return induce(P, Set(layers))
end

function bottom_layers(P::SimplePoset)
    bottom_card = minimum(cardinality.(elements(P)))
    layers = filter(x -> cardinality(x) <= bottom_card + 1, elements(P))
    return induce(P, Set(layers))
end

function proba_upper_Th(x::String, P::SimplePoset, h::Int, k::Int, I::Vector{String})
    II = isolated(induce(P, Set(filter(!=(x), elements(P)))))
    III = setdiff(II, I)
    return (1 / h) * (prod([h - 1 + k - length(II) + i for i in 1:length(II)])) / (prod([h - 1 + k - length(II) + i for i in 1:length(II)]) + length(I) * prod([h - 1 + k - length(II) + i for i in 1:length(III)]) * prod([h + k - length(I) + i for i in 1:(length(I)-1)]))
end

function proba_lower_Th(x::String, P::SimplePoset, h::Int, k::Int, I::Vector{String})
    II = isolated(induce(P, Set(filter(!=(x), elements(P)))))
    III = setdiff(II, I)
    return (prod([h - 1 + k - length(II) + i for i in 1:length(III)]) * prod([h + k - length(I) + i for i in 1:(length(I)-1)])) / (prod([h - 1 + k - length(II) + i for i in 1:length(II)]) + length(I) * prod([h - 1 + k - length(II) + i for i in 1:length(III)]) * prod([h + k - length(I) + i for i in 1:(length(I)-1)]))
end

function proba_upper_Bh(x::String, P::SimplePoset, h::Int, k::Int, I::Vector{String})
    II = isolated(induce(P, Set(filter(!=(x), elements(P)))))
    III = setdiff(II, I)
    return (prod([h - length(II) + k - 1 + i for i in 1:length(III)]) * prod([h - length(I) + k + i for i in 1:(length(I)-1)])) / (prod([h - length(II) + k - 1 + i for i in 1:length(II)]) + length(I) * prod([h - length(II) + k - 1 + i for i in 1:length(III)]) * prod([h - length(I) + k + i for i in 1:(length(I)-1)]))
end

function proba_lower_Bh(x::String, P::SimplePoset, h::Int, k::Int, I::Vector{String})
    II = isolated(induce(P, Set(filter(!=(x), elements(P)))))
    III = setdiff(II, I)
    return (1 / k) * (prod([h - length(II) + k - 1 + i for i in 1:length(II)])) / (prod([h - length(II) + k - 1 + i for i in 1:length(II)]) + length(I) * prod([h - length(II) + k - 1 + i for i in 1:length(III)]) * prod([h - length(I) + k + i for i in 1:(length(I)-1)]))
end

function generate_linext(P::SimplePoset, rng::AbstractRNG)
    H = deepcopy(P)
    lmin = String[]
    lmax = String[]

    while height(H) > 2
        if length(maximals(H)) == 1
            M = maximals(H)[1]
        else
            Th = top_layers(H)
            ul = upper_layer(Th)
            ll = lower_layer(Th)
            h = length(ul)
            k = length(ll)
            I = intersect(isolated(Th), ll)
            proba = [proba_upper_Th.(ul, Ref(Th), Ref(h), Ref(k), Ref(I)); proba_lower_Th.(I, Ref(Th), Ref(h), Ref(k), Ref(I))]
            M = sample(rng, [ul; I], ProbabilityWeights(proba))
        end
        pushfirst!(lmax, M)
        delete!(H, M)

        if length(minimals(H)) == 1
            m = minimals(H)[1]
        else
            Bh = bottom_layers(H)
            ul = upper_layer(Bh)
            ll = lower_layer(Bh)
            h = length(ul)
            k = length(ll)
            I = intersect(isolated(Bh), ul)
            proba = [proba_lower_Bh.(ll, Ref(Bh), Ref(h), Ref(k), Ref(I)); proba_upper_Bh.(I, Ref(Bh), Ref(h), Ref(k), Ref(I))]
            m = sample(rng, [ll; I], ProbabilityWeights(proba))
        end
        push!(lmin, m)
        delete!(H, m)
    end

    while height(H) == 2
        ul = upper_layer(H)
        ll = lower_layer(H)
        h = length(ul)
        k = length(ll)
        if h <= k
            if length(maximals(H)) == 1
                M = maximals(H)[1]
            else
                I = intersect(isolated(H), ll)
                proba = [proba_upper_Th.(ul, Ref(H), Ref(h), Ref(k), Ref(I)); proba_lower_Th.(I, Ref(H), Ref(h), Ref(k), Ref(I))]
                M = sample(rng, [ul; I], ProbabilityWeights(proba))
            end
            pushfirst!(lmax, M)
            delete!(H, M)
        else
            if length(minimals(H)) == 1
                m = minimals(H)[1]
            else
                I = intersect(isolated(H), ul)
                proba = [proba_lower_Bh.(ll, Ref(H), Ref(h), Ref(k), Ref(I)); proba_upper_Bh.(I, Ref(H), Ref(h), Ref(k), Ref(I))]
                m = sample(rng, [ll; I], ProbabilityWeights(proba))
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

seed!(parse(Int, ARGS[2]))

println(generate_linext(BooleanLattice(parse(Int, ARGS[1])), default_rng()))
