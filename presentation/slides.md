---
marp: true
headingDivider: 2
theme: custom
---

# Changements

## Performances des alternatives arrondies

Pour éviter plusieurs problèmes de précision flottante, les performances des alternatives et profils sont arrondies

## Prise en compte du préordre dans la fonction objectif

- DM1 : A = B > C = D = E
- DM2 : A = B = C > D = E
- Collectif : A = B = C = D = E

1 seul changement de comparaison pour chaque DM
Mais 6 changements de comparaisons au niveau du préordre pour chaque DM

- Meilleur collectif : A = B > C = D = E (ou A = B = C > D = E)

4 changements de comparaisons au niveau du préordre pour un seul DM
