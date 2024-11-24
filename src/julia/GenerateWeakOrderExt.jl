using Combinatorics
using Logging
using Random
using SimplePosets
using StatsBase


# Utility

function get_index(A, x)
    return findfirst(==(x), A)
end


# Conversion

function set2digits(A, B)
    return parse(UInt128, join(UInt8.(x ∈ A for x in B)), base=2)
end


# Bitwise operations

function set_diff(a, b)
    return a & ~b
end

function is_subset(a, b)
    return (a | b) == b
end


# Triangular matrix

function index(r, c, size)
    return size * (r - 1) - r * (r - 1) ÷ 2 + c - r
end

function slice(r, size)
    offset = size * (r - 1) - r * (r - 1) ÷ 2
    return offset+1:offset+size-r
end


# Poset basics

function ideal(P, A)
    return filter(y -> (y ∈ A) || (any(has(P, y, x) for x ∈ A)), elements(P))
end

function cover(P, x, y)
    return has(P, x, y) && isempty(interval(P, x, y))
end

function succ(P, x::String)
    return filter(y -> cover(P, x, y), elements(P))
end

function succ(P, A::Vector{String})
    return reduce(union, succ(P, x) for x ∈ A)
end


# AllWeak3

function AllWeak3!(labels, P, Y, A, base)
    if isempty(Y)
        return
    end

    for B ∈ powerset(Y, 1)
        AA_label = ideal(P, A) ∪ B
        AA = maximals(induce(P, Set(AA_label)))
        AA_digit = set2digits(AA_label, base)
        if !insorted(AA_digit, labels)
            insert!(labels, searchsortedfirst(labels, AA_digit), AA_digit)

            @debug "$(length(labels))"

            YY = minimals(induce(P, Set(setdiff(Y, B) ∪ succ(P, B))))
            AllWeak3!(labels, P, YY, AA, base)
        end
    end

    return
end


# generate_WE

function generate_WE(P::SimplePoset{T}) where {T}
    labels = zeros(UInt128, 1)
    @debug "start WE"
    AllWeak3!(labels, P, minimals(P), T[], reverse(elements(P)))
    @debug "end WE"

    @debug "start traversal"
    nb_paths = ones(UInt128, 1)
    NV = length(labels)
    for (i, u) ∈ pairs(reverse(labels[begin:end-1]))
        pushfirst!(nb_paths, sum(nb_paths[is_subset.(Ref(u), labels[NV - i + 1:end])]))
        
        @debug "$(length(nb_paths)) / $NV"
    end
    @debug "end traversal"

    return labels, nb_paths
end


# generate_weak_order_ext

function generate_weak_order_ext(labels, nb_paths, subsets, rng=Random.default_rng())
    result = Vector{String}[]
    N = length(labels)

    u = 1
    while u != N
        Nu = ((u+1):N)[is_subset.(Ref(labels[u]), labels[(u+1):N])]
        v = sample(rng, Nu, FrequencyWeights(nb_paths[Nu], nb_paths[u]))
        push!(result, subsets[digits(Bool, set_diff(labels[v], labels[u]), base=2, pad=length(subsets))])
        u = v
    end

    return result
end

