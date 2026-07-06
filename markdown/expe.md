# Expériences à mener pour le 3e chapitre de contributions

## Ordre d'exécution

- Tester SA de groupe *(fait)*
- Hyperparamétrage de SA *(en cours)*
- Tester plusieurs modèles de groupe possibles pour le calcul des chemins de préférences
- Mettre les deux ensemble

## Configuration de paramètres à tester

- **Nb de décideurs :** 2, 4, 6 (dépend du nombre de coeurs en parallèle utilisés)
- **Nb d'alternatives :** 100 (ou 500 si motivé) (rend SA et chemins de préférences plus long, car mouvements plus petits sur les profils)
- **Nb critères :** 3, 4, 5, 6 (RMP, car limité par génération d'instances uniforme), plus pour SRMP (pourquoi pas jusqu'à 15, mais rend tous les aspects du processus plus long)
- **Nb profils :** 1, 2, 3 (pareil, rend SA et chemins de préférences plus long)
- **Nb comparaisons d'entraînement :** 50, 100, 300, 500, 1000 (rend SA et chemins de préférences plus difficile à terminer)
- 2 (ou 3) **types de groupes :** proches mais peu enclin au compromis, éloignés mais faciles à convaincre (proches et simples à convaincre)
- **Nb de modèles de groupe possibles à chaque itérations :** 2, 3, 4, 5, 6 (dépend du temps de calcul et de leur diversité)

3 x 4 x 3 x 5 x 2 x 5 = 1800 configurations possibles

Notes :

sections sur l'extension du SA au cas multi décideur (plus précis)

tester cas extrêmes sur les paramètres

comparer SA et MILP sur les 2 améliorations (une partie à part des résultats)