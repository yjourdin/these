module Bit

export decode, encode

using .Iterators: takewhile, filter, map
using IterTools: iterated

# Set operations

const empty                 = typemin(UInt128)
const full                  = typemax(UInt128)
length(a)                   = ndigits(a; base = 2)
isempty(a)                  = a == empty
union(a, b)                 = a | b
union(itr; init = empty)    = reduce(|, itr; init = init)
intersect(a, b)             = a & b
intersect(itr; init = full) = reduce(&, itr; init = init)
setdiff(a, b)               = a & ~b
symdiff(a, b)               = a ⊻ b
isdisjoint(a, b)            = isempty(intersect(a, b))
issubset(a, b)              = intersect(a, b) == a
in(x, a)                    = issubset(encode(x), a)
in(a)                       = Base.Fix2(in, a)
subsets(a)                  = takewhile(!isempty, iterated(x -> (x - 1) & a, a))

# Encode / Decode

encode(x::Integer) = one(UInt128) << (x - 1)
encode(A) = sum(encode, A; init = empty)
decode(c) = filter(in(c), 1:length(c))

end
