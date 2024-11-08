using Combinatorics
using Graphs
using Logging
using Mmap
using Random
using SimplePosets
using StatsBase


# Utility

function get_index(A, x)
    return findfirst(==(x), A)
end


# Conversion

function set2digits(A, B)
    return parse(Int128, join(Int.(x ∈ A for x in B)), base=2)
end


# Digits bitwise operations

function set_diff(a, b)
    return a & ~b
end


# Triangular matrix

function index(r, c, size)
    return size * (r - 1) - r * (r - 1) ÷ 2 + c - r
end

function slice(r, size)
    offset = size * (r - 1) - r * (r - 1) ÷ 2
    return offset+1:offset+size-r
end


# Poset basics

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

function AllWeak3!(WE, vertex_labels, edge_labels_dict, subsets, P, Y, A)
    if isempty(Y)
        return
    end

    A_digit = set2digits(A, subsets)
    A_index = get_index(vertex_labels, A_digit)::Int
    for B in powerset(Y, 1)
        @debug length(vertex_labels)
        AA = maximals(induce(P, Set(ideal(P, A) ∪ B)))
        AA_digit = set2digits(AA, subsets)
        AA_index = get_index(vertex_labels, AA_digit)
        visited = true
        if isnothing(AA_index)
            visited = false
            add_vertex!(WE)
            push!(vertex_labels, AA_digit)
            AA_index = length(vertex_labels)
        end
        add_edge!(WE, A_index, AA_index)
        edge_labels_dict[(A_index, AA_index)] = set_diff(AA_digit, A_digit)

        if !visited
            YY = minimals(induce(P, Set(setdiff(Y, B) ∪ succ(P, B))))
            AllWeak3!(WE, vertex_labels, edge_labels_dict, subsets, P, YY, AA)
        end
    end
    return
end


# generate_WE

function generate_WE(nb_paths_io, edge_labels_io, P::SimplePoset{T}, logging_io=nothing) where {T}
    if !isnothing(logging_io)
        logger = SimpleLogger(logging_io, Debug)
        global_logger(logger)
    end

    WE = SimpleDiGraph(1)
    subsets = elements(P)
    vertex_labels = [set2digits(T[], subsets)]
    edge_labels_dict = Dict{Tuple{Int,Int},Int128}()

    @debug "start WE"
    AllWeak3!(WE, vertex_labels, edge_labels_dict, subsets, P, minimals(P), T[])
    @debug "end WE"

    @debug "start TS"
    ts = topological_sort(WE)
    ts_index = similar(ts)
    for (i, u) ∈ pairs(ts)
        ts_index[u] = i
    end
    @debug "end TS"

    @debug "start copy"
    N = nv(WE)
    nb_paths = mmap(nb_paths_io, Vector{Int128}, N)
    edge_labels = mmap(edge_labels_io, Vector{Int128}, N * (N - 1) ÷ 2)
    for ((u, v), d) ∈ pairs(edge_labels_dict)
        edge_labels[index(ts_index[u], ts_index[v], N)] = d
    end
    Mmap.sync!(edge_labels)
    @debug "end copy"

    @debug "start traversal"
    L = length(ts)
    for (i, u) ∈ pairs(reverse(ts))
        @debug "$i / $L"

        Nu = outneighbors(WE, u)

        if isempty(Nu)
            nb_paths[ts_index[u]] = big(1)
        else
            for v ∈ Nu
                for w ∈ outneighbors(WE, v)
                    if !has_edge(WE, u, w)
                        add_edge!(WE, u, w)
                        edge_labels[index(ts_index[u], ts_index[w], N)] = edge_labels[index(ts_index[u], ts_index[v], N)] | edge_labels[index(ts_index[v], ts_index[w], N)]
                    end
                end
            end

            nb_paths[ts_index[u]] = sum(nb_paths[ts_index[outneighbors(WE, u)]])
        end
        Mmap.sync!(nb_paths)
        Mmap.sync!(edge_labels)
    end
    @debug "end traversal"

    return WE
end


# generate_weak_order_ext

function generate_weak_order_ext(nb_paths_io, edge_labels_io, subsets, rng=Random.default_rng())
    result = Vector{String}[]

    nb_paths = mmap(nb_paths_io, Vector{Int128})
    edge_labels = mmap(edge_labels_io, Vector{Int128})
    N = length(nb_paths)

    u = 1
    s = slice(u, N)

    while !isempty(s)
        Nu = findall(!=(0), edge_labels[s]) .+ u
        v = sample(rng, Nu, FrequencyWeights(nb_paths[Nu], nb_paths[u]))
        push!(result, subsets[reverse(digits(Bool, edge_labels[index(u, v, N)], base=2, pad=length(subsets)))])

        u = v
        s = slice(u, N)
    end

    return result
end

