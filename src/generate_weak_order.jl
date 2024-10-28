import FromFile: @from

using Random
@from "GenerateWeakOrder.jl" using GenerateWeakOrder


Random.seed!(parse(Int, ARGS[2]))
println(random_ranking(parse(Int, ARGS[1])))

# @time println(random_ranking(parse(Int, "10")))
# @profview println(random_ranking(parse(Int, "10")))