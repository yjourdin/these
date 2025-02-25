# Strategies to choose pairwise comparisons to change

## Basic

- **Minimal number changes strategy :** Plus petit ensemble de paires à changer
  - Décision la plus "facile" à prendre pour améliorer le consensus
- **Maximal number changes strategy :** Plus grand ensemble de paires à changer
  - Décision la plus "efficace" pour améliorer le consensus, si acceptée
- **Least DM change strategy :** Paires sur lesquelles le plus de décideurs sont d'accord (avec quand même au moins un décideur différent)
- **Most DM change strategy :** Paires sur lesquelles le plus de décideurs doivent changer leur préférence (pour converger vers la majorité)
- **Equity DM change strategy :** Paires qui changent les préférences des décideurs ayant le moins changé leurs préférences jusqu'alors
  - Décision la plus "équitable" à court terme

<!-- ## Consistency

- **Maximal consistency strategy :** Paire qui change les préférences des décideurs ayant la meilleure cohérence entre son modèle et ses préférences
  - Similaire à **Equity change strategy** -->

## Consensus

> Soit la préférence du décideur $h$ sur la paire $(i, j)$ : $p^h_{i, j} = \begin{cases} 1 & \text{si } i \succ j \\ 0.5 & \text{si } i \sim j \\ 0 & \text{si } i \prec j \end{cases}$
>
> Soit la mesure de similarité entre deux décideurs $h$ et $l$ sur la paire $(i, j)$ : $sm^{h, l}_{i, j} = 1 - \lvert p^h_{i, j} - p^l_{i, j} \rvert \in [0, 1]$
>
> Soit le consensus entre les décideurs sur la paire $(i, j)$ : $cop_{i, j} = \frac{2}{n(n-1)}\sum_{h \ne l} sm^{h, l}_{i, j}$
>
> [1] E. Herrera-Viedma, S. Alonso, F. Chiclana, et F. Herrera, « A Consensus Model for Group Decision Making With Incomplete Fuzzy Preference Relations », IEEE Transactions on Fuzzy Systems, vol. 15, nᵒ 5, p. 863‑877, oct. 2007, doi: 10.1109/TFUZZ.2006.889952.

## SRMP parameters

- **Closer profile strategy :** Paire qui permet de rapprocher le plus possible les profils des décideurs vers les profils du modèle de compromis
- **Closer weight strategy :** Paire qui permet de rapprocher le plus possible les poids des décideurs vers les poids du modèle de compromis
- **Closer SRMP parameter strategy :** Paire qui permet de rapprocher le plus possible les paramètres des modèles des décideurs vers ceux du modèle de compromis
- **Closer SRMP strategy :** Paire qui permet de rapprocher le plus possible les modèles des décideurs vers le modèle de compromis (au moyen d'une distance custom)
- **One more SRMP parameter strategy :** Paires qui permettent de fixer en commun entre tous les décideurs un paramètre du modèle de compromis

- **Minimal consensus pair strategy :** Paire qui a le plus faible consensus

## Graph

- **Precedence pair strategy :** Identifie les ensembles de paires modifiés en bougeant les paramètres des modèles, et proposer les ensembles de paires les plus petits

<!-- Problème : changer qu'une seule relation peut transformer des préférences SRMP-compatible en non-compatible. -->