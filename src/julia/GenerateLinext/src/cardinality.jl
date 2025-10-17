cardinality(i)        = count_ones(i)

top_cardinality(A)    = maximum(cardinality, A)
bottom_cardinality(A) = minimum(cardinality, A)

layer(A, card) = findall(@. cardinality(A) == card)
