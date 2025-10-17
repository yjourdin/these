function generate_WE(P)
    BP = BitPoset(P)
    labels = [Bit.empty]

    AllWeak3!(labels, BP, min(BP), Bit.empty)

    NV       = length(labels)
    nb_paths = ones(UInt128, NV)
    for i ∈ (NV - 2):-1:1
        nb_paths[i] = sum(x -> nb_paths[x], successors(labels, i); init = UInt128(0))
        # @info "Vertices traversed : $(length(nb_paths) - i + 1) / $(length(nb_paths))"
    end

    return WE(labels, nb_paths)
end