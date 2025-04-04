using Combinatorics
using Logging
using Random
using SimplePosets
using StatsBase

# Utility

get_index(A, x) = findfirst(==(x), A)

# Conversion

set2digit(set, base) = parse(UInt128, join(UInt8.(x ∈ set for x ∈ base)); base = 2)

digit2set(digit, base) = base[digits(Bool, digit; base = 2, pad = length(base))]

# Bitwise operations

set_diff(a, b) = a & ~b

is_subset(a, b) = (a | b) == b

# Triangular matrix

index(r, c, size) = size * (r - 1) - r * (r - 1) ÷ 2 + c - r

function slice(r, size)
    offset = size * (r - 1) - r * (r - 1) ÷ 2
    return (offset + 1):(offset + size - r)
end

# Poset basics

ideal(P, A) = filter(y -> (y ∈ A) || (any(has(P, y, x) for x ∈ A)), elements(P))

cover(P, x, y) = has(P, x, y) && isempty(interval(P, x, y))

succ(P, x::String) = filter(y -> cover(P, x, y), elements(P))

succ(P, A::Vector{String}) = reduce(union, succ(P, x) for x ∈ A)

# AllWeak3

function AllWeak3!(labels, P, Y, A, base)
    if isempty(Y)
        return
    end

    for B ∈ powerset(Y, 1)
        AA_label = ideal(P, A) ∪ B
        AA = maximals(induce(P, Set(AA_label)))
        AA_digit = set2digit(AA_label, base)
        if !insorted(AA_digit, labels)
            insert!(labels, searchsortedfirst(labels, AA_digit), AA_digit)

            @debug "Vertices created : $(length(labels))"

            YY = minimals(induce(P, Set(setdiff(Y, B) ∪ succ(P, B))))
            AllWeak3!(labels, P, YY, AA, base)
        end
    end

    return
end

# generate_WE

function generate_WE(P::SimplePoset{T}) where {T}
    labels = zeros(UInt128, 1)

    AllWeak3!(labels, P, minimals(P), T[], reverse(elements(P)))

    nb_paths = ones(UInt128, 1)
    NV = length(labels)
    for (i, u) ∈ pairs(reverse(labels[begin:(end - 1)]))
        pushfirst!(
            nb_paths,
            sum(nb_paths[is_subset.(Ref(u), labels[(NV - i + 1):end])]),
        )

        @debug "Vertices traversed : $(length(nb_paths)) / $NV"
    end

    return labels, nb_paths
end

# generate_weak_order_ext

function generate_weak_order_ext(labels, nb_paths, base, rng = Random.default_rng())
    result = Vector{String}[]
    N = length(labels)

    u = 1
    while u != N
        Nu = ((u + 1):N)[is_subset.(Ref(labels[u]), labels[(u + 1):N])]
        v = sample(rng, Nu, FrequencyWeights(nb_paths[Nu], nb_paths[u]))
        push!(result, digit2set(set_diff(labels[v], labels[u]), base))
        u = v
    end

    return result
end

# number_of_arcs

function number_of_arcs(labels)
    n = big(0)
    N = length(labels)

    for i ∈ 1:N
        @debug "$i / $N"
        for j ∈ (i + 1):N
            if is_subset(labels[i], labels[j])
                n += 1
            end
        end
    end

    return n
end