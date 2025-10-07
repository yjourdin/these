include("Bit.jl")

using .Bit
using Posets: Posets

function remove_vertex!(P, labels, v)
    Posets.rem_vertex!(P, v)

    labels[v], labels[end] = labels[end], labels[v]
    return pop!(labels)
end

Posets.subset_decode(c::Integer)::Set{Int} = Set{Int}(collect(decode(c - 1)))
Posets.subset_encode(A::Set{T} where {T <: Integer})::Int = Int(encode(A)) + 1

above(a, P) = Posets.above(P, a)
below(a, P) = Posets.below(P, a)
just_above(a, P) = Posets.just_above(P, a)
just_below(a, P) = Posets.just_below(P, a)