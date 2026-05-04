function number_of_arcs(labels)
    n = 0
    N = length(labels)

    for i ∈ 1:(N-1)
        # @debug "$i / $N"
        u = labels[i]
        for j ∈ (i+1):N
            v = labels[j]
            Bit.issubset(u, v) && (n += 1)
        end
    end

    return n
end