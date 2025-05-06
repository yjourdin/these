using Posets: Posets

function subset_decode(c)
    bits = digits(c; base = 2)
    return findall(bits .> 0)
end
Posets.subset_decode(c::Integer) = subset_decode(c - 1)

function subset_encode(A)
    s = UInt128(0)
    for k ∈ A
        s += UInt128(1) << (k - 1)
    end
    return s
end
Posets.subset_encode(A::Set{T}) where {T <: Integer} = subset_encode(A) + 1