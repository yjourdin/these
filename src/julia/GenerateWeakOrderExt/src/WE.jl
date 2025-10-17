@kwdef struct WE
    labels   :: Vector{UInt128}
    nb_paths :: Vector{UInt128}
end
WE(d) = WE(d["labels"], d["nb_paths"])