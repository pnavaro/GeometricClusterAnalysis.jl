export HClust

struct HClust

  couleurs :: Vector{Int}
  Couleurs :: Vector{Vector{Int}}
  Temps_step :: Vector{Float64}
  Naissance :: Vector{Float64}
  Mort :: Vector{Float64}
  Indices_depart :: Vector{Int}

end
