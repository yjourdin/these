@memoize function w(m, k)
    k > m && return big(0)
    k == 1 && return big(1)
    return k * (w(m - 1, k) + w(m - 1, k - 1))
end

W(m) = sum(w(m, k) for k ∈ 1:m)
