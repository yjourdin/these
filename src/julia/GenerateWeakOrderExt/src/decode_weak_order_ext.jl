function decode_weak_order_ext(we)
    result = Vector{Vector{Int}}[]
    for x ∈ we
        tmp = Vector{Int}[]
        for xx ∈ decode(x)
            @chain xx begin
                subset_decode
                collect
                push!(tmp, _)
            end
        end
        push!(result, tmp)
    end
    return result
end