function generate_WE(M)
    N          = DEDEKIND[M]
    P          = M |> subset_lattice |> BitPoset
    labels_set = Set{Bitset}([Bit.empty])
    sizehint!(labels_set, N)

    AllWeak3!(labels_set, P, min(P), Bit.empty)

    labels   = sort!(collect(labels_set))
    nb_paths = ones(UInt128, N)
    for i ∈ (N - 1):-1:1
        u  = labels[i]
        nu = nb_paths[i]
        for j ∈ 1:(i - 1)
            Bit.issubset(labels[j], u) && (nb_paths[j] += nu)
        end
        # @info "Vertices traversed : $(length(nb_paths) - i + 1) / $(length(nb_paths))"
    end

    return WE(labels, nb_paths)
end