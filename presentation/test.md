---
headingDivider: 3
math: mathjax
marp: true
theme: custom
---

# Réunion hebdomadaire

Yann Jourdin

22 octobre

<!-- Notes -->

## Chemins entre 2 modèles RMP / SRMP

- Réutilisation des opérateurs de voisinages du recuit simulé
- Adaptation pour avoir des changements locaux
  - Profils entre 2 alternatives
  - Valeurs de poids occasionnant un changement
  - etc

## Génération uniforme de modèles RMP

Génération uniforme de **relation d'importance**

Génération uniforme de capacité $\to$ relation d'importance

*Problème :* Génération uniforme de capacité = Génération uniforme de relation d'importance ?

### Actuellement

> **Génération uniforme de capacité :**
>
> 1. Génère uniformément un ordre total respectant l'inclusion ($\{\} < \{1\} < \{2\} < \{1, 2\}$)
> 2. Génère $n$ réels $z_i \in [0, 1]$, les trie, et les assigne à chaque ensemble dans l'ordre (0.1, 0.3, 0.4, 0.9)

*Problème :* Très difficile d'avoir une équivalence entre 2 ensembles, il faut générer 2 réels égaux (presque impossible)

### Piste

> **Génération uniforme de préordre total :**
>
> - Génère uniformément un préordre total (sait faire)
> - jusqu'à trouver un qui est monotone au sens de l'inclusion

*Problème :* Très long à partir de 4 critères

### En cours

Trouver un papier qui génère uniformément un préordre total à partir d'un préordre (relation d'inclusion)

## Article étendu pour MIC

**Deadline :** 15 février 2025

**Extensions :**

- Hyperparamétrage (*à terminer*)
- Ajouter d'inconsistances dans l'ensemble d’entraînement (*à lancer*)
- Éliciter un modèle RMP à partir d'un SRMP, et vice-versa (*à lancer*)
- Trouver le nombre optimal de profils (*recherche autour d'un ou plusieurs opérateurs*)

Relancé des expériences, mais arrêt prématuré, investigations en cours

## Test colonnes

<div class="container">
<div class="col">

aze

</div>
<div class="col">

rty

</div>
</div>
