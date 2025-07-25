using Chain
using IterTools
using Logging
using Posets
using Random
using StatsBase

include("Posets.jl")

# Bitwise operations

bit_setdiff(a, b)  = a & ~b
bit_issubset(a, b) = (a & b) == a
bit_subsets(a)     = Iterators.takewhile(>(0), iterated(x -> (x - 1) & a, a))

# Poset basics

poset_ideal(A, P)      = (y for y ∈ 1:nv(P) if any(P[y] ≤ P[x] for x ∈ A))
poset_max(A, P)        = (x for x ∈ A if !any(P[x] < P[y] for y ∈ A))
poset_min(A, P)        = (x for x ∈ A if !any(P[y] < P[x] for y ∈ A))
poset_just_above(a, P) = just_above(P, a)

# AllWeak3

function AllWeak3!(labels, P, Y, A)
    Y == 0 && return

    ideal_A = @chain A begin
        subset_decode
        poset_ideal(P)
        subset_encode
    end

    for B ∈ bit_subsets(Y)
        A′ = ideal_A | B
        i = searchsortedfirst(labels, A′)
        get(labels, i, nothing) ≠ A′ || continue

        insert!(labels, i, A′)
        @debug "Vertices created : $(length(labels))"
        Y′ = @chain B begin
            subset_decode
            poset_just_above.(Ref(P))
            @. subset_encode
            reduce(|, _; init = bit_setdiff(Y, B))
            subset_decode
            poset_min(P)
            subset_encode
        end
        AllWeak3!(labels, P, Y′, A′)
    end

    return
end

# successors

function successors(labels, i)
    u = labels[i]
    S_labels = @view labels[(i + 1):end]
    S_indices = S_labels.indices[1]
    return (S_indices[j] for (j, v) ∈ pairs(S_labels) if bit_issubset(u, v))
end

# generate_WE

function generate_WE(P)
    labels = zeros(UInt128, 1)

    AllWeak3!(labels, P, subset_encode(minimals(P)), 0)

    NV       = length(labels)
    nb_paths = ones(UInt128, NV)
    for i ∈ (NV - 1):-1:1
        nb_paths[i] = sum(x -> nb_paths[x], successors(labels, i))
        @debug "Vertices traversed : $(length(nb_paths) - i + 1) / $(length(nb_paths))"
    end

    return labels, nb_paths
end

# generate_weak_order_ext

function generate_weak_order_ext(labels, nb_paths, rng = Random.default_rng())
    result = Vector{Vector{Int}}[]
    N      = length(labels)
    i      = 1
    while i ≠ N
        Ni = collect(successors(labels, i))
        j  = sample(rng, Ni, FrequencyWeights(view(nb_paths, Ni), nb_paths[i]))
        @chain labels begin
            bit_setdiff(_[j], _[i])
            subset_decode
            @. Posets.subset_decode
            push!(result, _)
        end
        i = j
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
            bit_issubset(labels[i], labels[j]) && (n += 1)
        end
    end

    return n
end

# WE

@kwdef struct WE
    labels   :: Vector{UInt128}
    nb_paths :: Vector{UInt128}
end

WE(d) = WE(d["labels"], d["nb_paths"])