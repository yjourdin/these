module Posets

include("Bit.jl")
using .Bit: decode, encode

using Posets
import Posets: above, below, just_above, just_below

function remove_vertex!(P, labels, v)
    rem_vertex!(P, v)

    labels[v], labels[end] = labels[end], labels[v]
    return pop!(labels)
end

subset_decode(c) = decode(c - 1)
subset_encode(A) = encode(A) + 1

above(a, P) = above(P, a)
below(a, P) = below(P, a)
just_above(a, P) = just_above(P, a)
just_below(a, P) = just_below(P, a)

end # module Posets