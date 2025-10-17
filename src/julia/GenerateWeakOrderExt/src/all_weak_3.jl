function AllWeak3!(labels, P, Y, A)
    Bit.isempty(Y) && return

    ideal_A = ideal(A, P)

    for B ∈ Bit.subsets(Y)
        A′ = Bit.union(ideal_A, B)
        i = searchsortedfirst(labels, A′)
        get(labels, i, nothing) ≠ A′ || continue

        insert!(labels, i, A′)
        # @debug "Vertices created : $(length(labels))"
        Y′ = @chain B begin
            filter(P)
            Bit.union(Y)
            Bit.setdiff(B)
            min(P)
        end
        AllWeak3!(labels, P, Y′, A′)
    end

    return
end