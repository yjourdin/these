function proba_upper_Th(h, k, I, II, III)
    return (1 / h) * (prod(range(h - 1 + k - II + 1; length = II); init = big(1))) / (
        prod(range(h - 1 + k - II + 1; length = II); init = big(1)) +
        I *
        prod(range(h - 1 + k - II + 1; length = III); init = big(1)) *
        prod(range(h + k - I + 1; length = I - 1); init = big(1))
    )
end

function proba_lower_Th(h, k, I, II, III)
    return (
        prod(range(h - 1 + k - II + 1; length = III); init = big(1)) *
        prod(range(h + k - I + 1; length = I - 1); init = big(1))
    ) / (
        prod(range(h - 1 + k - II + 1; length = II); init = big(1)) +
        I *
        prod(range(h - 1 + k - II + 1; length = III); init = big(1)) *
        prod(range(h + k - I + 1; length = I - 1); init = big(1))
    )
end

function proba_upper_Bh(h, k, I, II, III)
    return (
        prod(range(h - II + k - 1 + 1; length = III); init = big(1)) *
        prod(range(h - I + k + 1; length = I - 1); init = big(1))
    ) / (
        prod(range(h - 1 + k - II + 1; length = II); init = big(1)) +
        I *
        prod(range(h - II + k - 1 + 1; length = III); init = big(1)) *
        prod(range(h - I + k + 1; length = I - 1); init = big(1))
    )
end

function proba_lower_Bh(h, k, I, II, III)
    return (1 / k) * (prod(range(h - II + k - 1 + 1; length = II); init = big(1))) / (
        prod(range(h - II + k - 1 + 1; length = II); init = big(1)) +
        I *
        prod(range(h - II + k - 1 + 1; length = III); init = big(1)) *
        prod(range(h - I + k + 1; length = I - 1); init = big(1))
    )
end

function proba_Th(h, k, I, II, III)
    eu    = prod(range(h - 1 + k - II + 1; length = II); init = big(1))
    el    = prod(range(h - 1 + k - II + 1; length = III); init = big(1)) * prod(range(h + k - I + 1; length = I - 1); init = big(1))
    denom = eu + I * el
    pu    = eu / (h * denom)
    pl    = el / denom
    return pu, pl
end

function proba_Bh(h, k, I, II, III)
    el    = prod(range(h - 1 + k - II + 1; length = II); init = big(1))
    eu    = prod(range(h - 1 + k - II + 1; length = III); init = big(1)) * prod(range(h + k - I + 1; length = I - 1); init = big(1))
    denom = eu + I * el
    pl    = el / (k * denom)
    pu    = eu / denom
    return pl, pu
end
