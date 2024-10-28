include("GenerateLinext.jl")

using Random
using SimplePosets
using .GenerateLinext

Random.seed!(parse(Int, ARGS[2]))
println(generate_linext(BooleanLattice(parse(Int, ARGS[1])), Random.default_rng()))