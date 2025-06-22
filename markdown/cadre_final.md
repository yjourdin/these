# Cadre de la recherche actuelle

## Meta

1. Les DMs veulent arriver à une décision commune
2. Les DMs acceptent de modifier leurs préférences pour arriver à une décision commune
3. Les DMs ont la même importance
4. Les DMs acceptent de travailler sur le même ensemble de critères (mais on pourrait modéliser des DMs qui ne considère pas un critère en lui attribuant un poids nul)
5. Les DMs travaillent sur les mêmes alternatives (difficile à justifier une décision commune sinon)

## Méthodologie

1. Les préférences des décideurs, ainsi que du groupe, sont modélisables par un même modèle, paramétrés différemment en fonction des DMs
2. Agrégation des modèles pour n'avoir qu'un modèle collectif (plutôt qu'agrégation des recommendations)

## Inférence

1. Élicitation à partir de paires de comparaisons
2. Possibilité d'ajouter des informations sur les paramètres (intervalles sur les profils, sur les poids, préordre sur les poids)
3. **Mise en oeuvre :** En considérant le modèle **SRMP**, déterminer à l'aide d'un **MILP** un modèle collectif, et déterminer des **chemins de préférences** pour y arriver

## Tests

1. Inconsistance (ou non)
2. Les comparaisons se font sur les mêmes paires d'alternatives (ou non)
