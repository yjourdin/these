include("Posets.jl")

using Chain
using Logging
using Random
using StatsBase

# Poset basics

poset_ideal(A, P)  = (y for y ∈ 1:nv(P) if any(P[y] ≤ P[x] for x ∈ A))
poset_filter(A, P) = (y for y ∈ 1:nv(P) if any(P[x] ≤ P[y] for x ∈ A))
poset_max(A, P)    = (x for x ∈ A if !any(P[x] < P[y] for y ∈ A))
poset_min(A, P)    = (x for x ∈ A if !any(P[y] < P[x] for y ∈ A))

# Bit poset

struct BitPoset
    in  :: Vector{UInt128}
    out :: Vector{UInt128}
end
BitPoset(P) = BitPoset(encode.(P.d.badjlist), encode.(P.d.fadjlist))

bit_poset_ideal(a, P)  = Bit.union(P.in[x] for x ∈ decode(a); init = a)
bit_poset_filter(a, P) = Bit.union(P.out[x] for x ∈ decode(a); init = a)
bit_poset_max(a, P)    = encode(x for x ∈ decode(a) if Bit.isdisjoint(P.out[x], a))
bit_poset_min(a, P)    = encode(x for x ∈ decode(a) if Bit.isdisjoint(P.in[x], a))

# AllWeak3

function AllWeak3!(labels, P, Y, A)
    Bit.isempty(Y) && return

    ideal_A = bit_poset_ideal(A, P)

    for B ∈ Bit.subsets(Y)
        A′ = Bit.union(ideal_A, B)
        i = searchsortedfirst(labels, A′)
        get(labels, i, nothing) ≠ A′ || continue

        insert!(labels, i, A′)
        # @debug "Vertices created : $(length(labels))"
        Y′ = @chain B begin
            bit_poset_filter(P)
            Bit.union(Y)
            Bit.setdiff(B)
            bit_poset_min(P)
        end
        AllWeak3!(labels, P, Y′, A′)
    end

    return
end

# successors

function successors(labels, i)
    u    = labels[i]
    succ = (i + 1):length(labels)
    u ≤ 1 && return succ
    return (j for j ∈ succ if Bit.issubset(u, labels[j]))
end

# generate_WE

function generate_WE(P)
    labels = [Bit.empty]

    AllWeak3!(labels, BitPoset(P), encode(Posets.minimals(P)), Bit.empty)

    NV       = length(labels)
    nb_paths = ones(UInt128, NV)
    for i ∈ (NV - 2):-1:1
        nb_paths[i] = sum(x -> nb_paths[x], successors(labels, i); init = UInt128(0))
        # @debug "Vertices traversed : $(length(nb_paths) - i + 1) / $(length(nb_paths))"
    end

    return labels, nb_paths
end

# generate_weak_order_ext

function generate_weak_order_ext(labels, nb_paths, rng = Random.default_rng())
    result = Vector{Vector{Int}}[]
    N      = length(labels)
    i      = 1
    while i ≠ N
        Ni       = collect(successors(labels, i))
        @views j = sample(rng, Ni, FrequencyWeights(nb_paths[Ni], nb_paths[i]))
        @chain labels begin
            Bit.setdiff(_[j], _[i])
            decode
            collect
            @. Posets.subset_decode
            @. collect
            push!(result, _)
        end
        i = j
    end
    return result
end

# number_of_arcs

function number_of_arcs(labels)
    n = 0

    for i ∈ eachindex(labels)
        # @debug "$i / $(length(labels))"
        for _ ∈ successors(labels, i)
            n += 1
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