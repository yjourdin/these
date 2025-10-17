is_isolated_top(P, x)    = !any(P[x] < P[y] for y ∈ 1:nv(P))
is_isolated_bottom(P, x) = !any(P[y] < P[x] for y ∈ 1:nv(P))

isolated_top(P, A)       = [x for x ∈ A if is_isolated_top(P, x)]
isolated_bottom(P, A)    = [x for x ∈ A if is_isolated_bottom(P, x)]
