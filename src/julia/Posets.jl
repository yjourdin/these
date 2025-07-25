using Posets: Posets

function remove_vertex!(P, labels, v)
    rem_vertex!(P, v)

    labels[v], labels[end] = labels[end], labels[v]
    return pop!(labels)
end

subset_decode(c) = findall(digits(Bool, c; base = 2))
Posets.subset_decode(c::Integer) = subset_decode(c - 1)

subset_encode(A) = sum(x -> 2^(x - 1), A; init = UInt128(0))
Posets.subset_encode(A::Set{T}) where {T <: Integer} = subset_encode(A) + 1