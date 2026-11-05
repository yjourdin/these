# Changement acceptation

## Avant

Quand une modification de ses préférences est proposée à un décideur, on considère l'ensemble des comparaisons intégrant cette modification $\mathcal{L}_i$.

Si il existe un modèle $\mathcal{M}'$ qui est "proche" en termes de paramètres du modèle du décideur $\mathcal{M}_i$, alors la modification est acceptée.

## Après

Quand une modification de ses préférences est proposée à un décideur, on considère uniquement les comparaisons modifiées $\mathcal{l}_i$.

À chaque décideur est associé une liste de comparaisons qu'il n'accepte pas $\mathcal{R}_i$.

Si une des nouvelles comparaisons $\mathcal{l}_i$ est dans la liste $\mathcal{R}_i$, alors elle est refusée.

Cette liste $\mathcal{R}_i$ est déterminée à partir de la "crédibilité" (puissance d'argument) des préférences initiales du décideur $\mathcal{L}$:

- Si $a \succ b \in \mathcal{L}$:

    SRMP : $r(a \succ b) = \sum_{a_j > p_j} w_j - \sum_{b_j > p_j} w_j$, pour le profil $p$ discriminant $a$ et $b$.

    RMP : $r(a \succ b) = \mu(\{j : a_j > p_j\}) - \mu(\{j : b_j > p_j\})$, pour le profil $p$ discriminant $a$ et $b$.

- Si $a \sim b \in \mathcal{L}$:

    SRMP : $r(a \succ b) = \min(\sum_{(a_j > p_j \text{ et } b_j > p_j) \text{ ou } (a_j < p_j \text{ et } b_j < p_j)} w_j, \quad p \in \mathcal{P})$.

    RMP : $r(a \succ b) = \min(\mu(\{j : (a_j > p_j \text{ et } b_j > p_j) \text{ ou } (a_j < p_j \text{ et } b_j < p_j)\}, \quad p \in \mathcal{P})$.

Soit un seuil $\lambda$, si $r(a \succ b) > \lambda$ (resp. $r(a \sim b) > \lambda$), alors on considère que le décideur a suffisamment d'arguments pour ne pas envisager un changement, sinon il peut envisager une modification de sa préférence entre $a$ et $b$.

## Modèle collectif acceptable par tous

Afin de savoir s'il existe un modèle de groupe qui convienne à tous les décideurs, il faut résoudre le MILP (celui qui élicite un modèle de groupe) en connaissant tous les comparaisons inenvisageables pour les décideurs $\mathcal{R}_i$. Si le MILP trouve une solution faisable, alors il existe un tel modèle collectif.