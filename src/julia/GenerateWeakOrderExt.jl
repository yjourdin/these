using Combinatorics
using Logging
using Random
using Posets
using StatsBase

include("Posets.jl")

# Bitwise operations

bit_setdiff(a, b)  = a & ~b
bit_issubset(a, b) = (a & b) == a

# Poset basics

poset_ideal(P, A) = (y for y ∈ 1:nv(P) if any(P[y] <= P[x] for x ∈ A))
poset_max(P, A)   = (x for x ∈ A if !any(P[x] < P[y] for y ∈ A))
poset_min(P, A)   = (x for x ∈ A if !any(P[y] < P[x] for y ∈ A))

# AllWeak3

function AllWeak3!(labels, P, Y, A)
    isempty(Y) && return

    ideal_A = poset_ideal(P, A)
    ideal_A_digit = subset_encode(ideal_A)

    Yset = BitSet(Y)
    for B ∈ powerset(Y, 1)
        B_digit = subset_encode(B)

        A′_digit = ideal_A_digit | B_digit
        if !insorted(A′_digit, labels)
            insert!(labels, searchsortedfirst(labels, A′_digit), A′_digit)

            @debug "Vertices created : $(length(labels))"

            YB    = setdiff(Yset, B)
            SuccB = Iterators.flatmap(Base.Fix1(just_above, P), B)
            Y′    = collect(poset_min(P, union!(YB, SuccB)))

            A′ = subset_decode(A′_digit)

            AllWeak3!(labels, P, Y′, A′)
        end
    end
    return
end

# generate_WE

function generate_WE(P)
    labels = zeros(UInt128, 1)

    AllWeak3!(labels, P, collect(minimals(P)), eltype(P)[])

    NV       = length(labels)
    nb_paths = ones(UInt128, NV)
    for (i, u) ∈ labels |> pairs |> Iterators.reverse |> Base.Fix2(Iterators.drop, 1)
        nb_paths[i] = sum(
            nb_paths[i + j] for
            (j, v) ∈ pairs(view(labels, (i + 1):NV)) if bit_issubset(u, v)
        )
        @debug "Vertices traversed : $(length(nb_paths) - i + 1) / $(length(nb_paths))"
    end

    return labels, nb_paths
end

# generate_weak_order_ext

function generate_weak_order_ext(labels, nb_paths, rng = Random.default_rng())
    result = Vector{Int}[]
    N      = length(labels)
    u      = 1
    while u != N
        Nu = [v for v ∈ (u + 1):N if bit_issubset(labels[u], labels[v])]
        v  = sample(rng, Nu, FrequencyWeights(view(nb_paths, Nu), nb_paths[u]))
        push!(result, subset_decode(bit_setdiff(labels[v], labels[u])) .- 1)
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
            bit_issubset(labels[i], labels[j]) && (n += 1)
        end
    end

    return n
end

# GraphFile

@kwdef struct GraphFile
    labels   :: Vector{UInt128}
    nb_paths :: Vector{UInt128}
end

GraphFile(d) = GraphFile(d["labels"], d["nb_paths"])