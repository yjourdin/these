# Chemins de préférences avec recuit simulé

## Nouveautés

- Élicitation du modèle de groupe avec un recuit simulé (plus rapide)
- Élicitation de plusieurs modèles de groupe (pour limiter les cas où le modèle de groupe trouvé est trop "loin" des modèles des décideurs)

## Hypothèses

- Phase d'élicitation du modèle de groupe plus rapide
- Modèle de groupe répondant moins à l'objectif d'équité entre décideurs
- Moins de cas avec chemins de préférences long à calculer (surtout dans les petites instances avec peu de critères et peu de paires de comparisons données par les décideurs)

## Mesures

- Temps de calcul (élicitation du groupe, et chemin de préférences)
- Nombre de comparisons modifiées par les décideurs (maximum entre les décideurs)
- Nombre d'itérations du processus (nombre d'élicitation du modèle de groupe, et nombre de retours vers les décideurs)

## À voir

- Nombre de modèles de groupe à éliciter en même temps donnant les meilleurs résultats
- Lancer plusieurs recuits à partir de plusieurs modèles de départ différents, plutôt que renvoyer l'ensembles des meilleurs modèles d'un recuit (pour plus de diversité de solutions), ou faire du reheating dans le recuit simulé