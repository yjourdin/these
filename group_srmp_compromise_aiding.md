# Group (S)RMP compromise aiding

## Profils

Chaque profil $p$ divise les paires en 3 ensembles :
- $c(a, p) \subset c(b, p) \implies a \succ b$ 
- $c(a, p) = c(b, p) \implies a \sim b$ 
- $c(a, p) \nsubseteq c(b, p) \implies a ? b$ (dépend des poids / relation d'importance sur les critères) 

Un profil suivant dans l'ordre lexicographique n'aura d'impact certain que sur les paires $\sim$ issues des profils précédents.

## Ordre lexicographique

Propose itérativement des profils aux décideurs, avec chaque profil proposé essayant de minimiser le nombre de paires contraires aux préférences, en montrant à chaque fois les paires qui seront fixées ($\succ$ et $\sim$)

*Étude sur le nombre de profils nécessaire pour avoir un modèle suffisament précis*

## Relation d'importance sur les critères

Propose aux décideurs itérativement des comparaisons de sous-ensembles de critères, en essayant de minimiser le nombre de paires d'alternatives contraires aux préférences, en montrant à chaque fois les paires qui seront fixées ($\succ$ et $\sim$)

## Poids sur les critères

Donne des indications sur les comparaisons entre les poids, en suivant le travail sur ELECTRE-Tri

OU

Propose aux décideurs 