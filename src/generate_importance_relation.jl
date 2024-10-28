import FromFile: @from

using Random
@from "GenerateImportanceRelation.jl" using GenerateImportanceRelation

Random.seed!(parse(Int, ARGS[2]))
println(generate_importance_relation(parse(Int, ARGS[1])))

# @time println(generate_importance_relation(parse(Int, "4")))
# @profview println(generate_importance_relation(parse(Int, "4")))