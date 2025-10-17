function successors(labels, i)
    u    = labels[i]
    succ = (i + 1):length(labels)
    u ≤ 1 && return succ
    return (j for j ∈ succ if Bit.issubset(u, labels[j]))
end