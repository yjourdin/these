function number_of_arcs(labels)
    n = 0

    for i ∈ eachindex(labels)
        # @debug "$i / $(length(labels))"
        for _ ∈ successors(labels, i)
            n += 1
        end
    end

    return n
end