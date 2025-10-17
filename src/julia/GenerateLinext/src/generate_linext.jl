function generate_linext!(P, rng = Random.default_rng())
    lmin        = Int[]
    lmax        = Int[]
    labels      = collect(0:(nv(P)-1))
    top_card    = top_cardinality(labels)
    bottom_card = bottom_cardinality(labels)

    while (length(labels) ≥ 2) && (top_card - bottom_card > 1)
        max = maximals(P)
        M, _ = iterate(max)
        if !isempty(max)
            ul = layer(labels, top_card)
            ll = layer(labels, top_card - 1)
            h  = length(ul)
            k  = length(ll)
            I  = isolated_top(P, ll)
            M  = select_M(P, ul, I, h, k, rng)
        end
        label = labels[M]
        pushfirst!(lmax, label)
        remove_vertex!(P, labels, M)
        (cardinality(label) ≠ top_card) || (top_card = top_cardinality(labels))

        min = minimals(P)
        m, _ = iterate(min)
        if !isempty(min)
            ul = layer(labels, bottom_card + 1)
            ll = layer(labels, bottom_card)
            h  = length(ul)
            k  = length(ll)
            I  = isolated_bottom(P, ul)
            m  = select_m(P, ll, I, h, k, rng)
        end
        label = labels[m]
        push!(lmin, label)
        remove_vertex!(P, labels, m)
        (cardinality(label) ≠ bottom_card) || (bottom_card = bottom_cardinality(labels))
    end

    while (length(labels) ≥ 2) && (top_card - bottom_card == 1)
        ul = layer(labels, top_card)
        ll = layer(labels, bottom_card)
        h  = length(ul)
        k  = length(ll)
        if h ≤ k
            max = maximals(P)
            M, _ = iterate(max)
            if !isempty(max)
                I = isolated_top(P, ll)
                M = select_M(P, ul, I, h, k, rng)
            end
            label = labels[M]
            pushfirst!(lmax, label)
            remove_vertex!(P, labels, M)
            (cardinality(label) ≠ top_card) || (top_card = top_cardinality(labels))
        else
            min = minimals(P)
            m, _ = iterate(min)
            if !isempty(min)
                I = isolated_bottom(P, ul)
                m = select_m(P, ll, I, h, k, rng)
            end
            label = labels[m]
            push!(lmin, label)
            remove_vertex!(P, labels, m)
            (cardinality(label) ≠ bottom_card) ||
                (bottom_card = bottom_cardinality(labels))
        end
    end

    append!(lmin, shuffle!(labels))

    return [lmin; lmax] .|> Bit.decode .|> collect
end
