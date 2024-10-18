# Stratégies

## Elicitation

### Consensus

- Nombre de comparaisons respectées
- Soit $p^h_{i, j} = \begin{cases} 1 & \text{si } i \succ j \\ 0.5 & \text{si } i \sim j \\ 0 & \text{si } i \prec j \end{cases}$, $\sum_{h, i, j} 1 - \lvert p^h_{i, j} - \bar{p}_{i, j} \rvert$ (somme pour chaque paire et chaque DM de la différence entre la préférence du DM et celle du groupe)  (prends en compte la différence entre $i \sim j$ et $i \prec j$ comparées à $i \succ j$)

### Group model

- Maximise le consensus total
- Maximise le consensus du "pire" DM
- Maximise le consensus pondéré pour chaque DM (un poids pour chaque DM)

### DM models

- Maximise le nombre de paramètres en commun avec le modèle de groupe (DM considéré comme SRMP-compatible)
- Maximise la "proximité" avec le modèle de groupe
- Maximise le nombre de paramètres étant les plus compliqués à modifier pour converger vers le modèle de groupe (ordre lexicographique -> poids -> profils)

## Preferences changes proposals

Pour chaque DM, déterminer une suite de changements des paramètres (un chemin) pour converger vers ceux du modèle de groupe, tout en ne modifiant pas les comparaisons du DM en accord avec le modèle de groupe

---

**À faire :** Prouver (ou tester) qu'un chemin peut être trouvé entre 2 modèles SRMP quelconques

---

**À vérifier :** Le modèle de groupe ne doit pas radicalement changer entre 2 itérations

---

**Extension :** Proposer plusieurs chemins

<!---
# Measures

- Consensus sur chaque paire d'alternative (nombre de DMs d'accord avec le modèle de groupe)
- Consensus entre les DMs (nombre de préférences différentes entre 2 décideurs)
- Proximité avec le modèle de groupe (nombre de préférences différentes avec le modèle de groupe pour chaque DM)
-->
