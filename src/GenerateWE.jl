using Combinatorics
using Graphs
using MetaGraphs
using SimplePosets


# Poset Basics

function ideal(P, A)
    return filter(y -> (y ∈ A) || (any(has(P, y, x) for x ∈ A)), elements(P))
end

function cover(P, x, y)
    return has(P, x, y) && isempty(interval(P, x, y))
end

function succ(P, x::String)
    return filter(y -> cover(P, x, y), elements(P))
end

function succ(P, A::Vector{String})
    return reduce(union, succ(P, x) for x ∈ A)
end


# AllWeak3

function AllWeak3!(WE, P, Y, A)
    if isempty(Y)
        return
    end

    for B in powerset(Y, 1)
        AA = maximals(induce(P, Set(ideal(P, A) ∪ B)))

        visited = true
        if !haskey(WE, AA, :name)
            visited = false
            add_vertex!(WE, :name, AA)
        end
        add_edge!(WE, WE[A, :name], WE[AA, :name], :label, setdiff(AA, A))

        if !visited
            YY = minimals(induce(P, Set(setdiff(Y, B) ∪ succ(P, B))))
            AllWeak3!(WE, P, YY, AA)
        end
    end
    return
end


# generate_WE

function generate_WE(P::SimplePoset{T}) where {T}
    WE = zero(MetaDiGraph())
    set_indexing_prop!(WE, :name)
    add_vertex!(WE, :name, T[])

    println("start WE")
    AllWeak3!(WE, P, minimals(P), T[])
    println(WE)
    println("end WE")

    println("start TS")
    rts = topological_sort(WE)
    reverse!(rts)
    println("end TS")

    println("start traversal")
    k = 0
    for u ∈ rts
        k+=1
        println(k, " | ", length(rts))
        Nu = outneighbors(WE, u)

        if isempty(Nu)
            set_prop!(WE, u, :nb_paths, big(1))
        else
            for v ∈ Nu
                for w ∈ setdiff(outneighbors(WE, v), Nu)
                    add_edge!(WE, u, w, :label, get_prop(WE, u, v, :label) ∪ get_prop(WE, v, w, :label))
                end
            end

            set_prop!(WE, u, :nb_paths, sum(v -> get_prop(WE, v, :nb_paths), outneighbors(WE, u)))
        end
    end
    println("end traversal")

    return WE
end
