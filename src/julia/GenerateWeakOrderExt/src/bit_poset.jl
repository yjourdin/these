struct BitPoset
    in  :: Vector{UInt128}
    out :: Vector{UInt128}
end
BitPoset(P) = BitPoset(encode.(P.d.badjlist), encode.(P.d.fadjlist))

ideal(a, P::BitPoset)  = Bit.union(P.in[x] for x ∈ decode(a); init = a)
filter(a, P::BitPoset) = Bit.union(P.out[x] for x ∈ decode(a); init = a)

max(P::BitPoset) = encode(x for (x, out) ∈ pairs(P.out) if Bit.isempty(out))
min(P::BitPoset) = encode(x for (x, in) ∈ pairs(P.in) if Bit.isempty(in))

max(a, P::BitPoset) = encode(x for x ∈ decode(a) if Bit.isdisjoint(P.out[x], a))
min(a, P::BitPoset) = encode(x for x ∈ decode(a) if Bit.isdisjoint(P.in[x], a))