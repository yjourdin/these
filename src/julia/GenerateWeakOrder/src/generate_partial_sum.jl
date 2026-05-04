const PartialSumType = BigFloat

function generate_partial_sum(m, delta = 0.01)
    Wm = W(m)
    k  = 0
    S  = PartialSumType[0]

    while (Wm - last(S) > delta) && ((length(S) < 2) || S[end] > S[end - 1])
        k += 1
        push!(S, last(S) + (big(k)^m) / (2^(k + 1)))
    end

    S[end] = Wm

    return S
end
