# EJOR response

## Contributions, présentation du problème, état de l'art, motivations

### Comparaisons manquantes par rapport aux *consensus-reaching processes* (CRP)

Il manquait la spécification des gaps. Par rapport aux CRP, ceux-ci ne produisent pas de modèle de groupe à la fin, ils ne se concentrent que sur les préférences exprimées.

### Il manque les gaps

Gaps :

- Soit pas de modèle de préférence de groupe en sortie (Consensus reaching process, ainsi que d'autres approches)
- Soit pas sur le problème de rangement (Cailloux + Damart sur ELECTRE Tri)

### Les contributions sont limitées (MILP déjà existant, et juste un GBFS)

Contributions relevées :

- fonction objectif du SRMP en minimax
- prend en compte les refus des décideurs entre les itérations (pas développé)
- ordonne les révisions par une recherche dans l'espace des modèles

En soi, la contribution générale est de proposer une séquence de changements à proposer aux décideurs en se plaçant dans l'espace des modèles, ainsi que comment prendre en compte les retours des décideurs dans l'inférence d'un modèle de groupe.

extension du MILP pour du groupe (à partir d'un MILP mono-décideur)

## Méthodologie

### Quel est l'intérêt des chemins de préférences au niveau des décideurs ?

Justifier le lien entre model-space et preference-space (et pourquoi on se met dans le model-space)
Garder une cohérence en termes de modèles que ce soit pour le classement de groupe ainsi que pour celui des décideurs

### Le modèle M_i n'est pas le modèle du décideurs

À partir des L_i, oui il existe plusieurs M_i compatible, et on ne trouvera pas forcément le bon M_i, du moins au début (cela se peaufinera au travers des retours des décideurs). Cependant, il faut réduire le nombre d'alternatives impliquées, car sinon l'ensemble des M_i est trop vaste et perds l'intérêt des chemins (trop facile à trouver)

### GBFS ne propose pas des chemins courts, et le pruning peut créer des gros changements de paramètres

Les affirmations comme quoi les chemins de préférences doivent être courts et proposer de petits chamngements en terme de comparaisons ne sont pas respecter par l'utilisation d'un GBFS ainsi que le pruning. Expliquer pourquoi on utilise un GBFS et pas un Astar à l'aide d'expériences ciblées, et préciser que le pruning évite de changer 2 fois la même comparaison.

### Le cas où un modèle n'est pas trouvé n'est pas décrit

Expliciter plus en profondeur ce cas-là
Alors il y a infaisabilité, et donc il faut en tirer des préférences qui entrent en contradiction à l'aide de méthodes adaptées (Mousseau)

### L'accumulation de refus peut mener à un problème infaisable, et si une étape de chemin est refusée, toutes les comparaisons ne sont pas forcéments refusées

Être plus clair sur le traitement du cas d'infaisabilité, ainsi que sur la prise en compte des refus par le MILP.
Si infaisabilité, alors cf au dessus
Dans le cas où des modifications sont refusées, toutes ne posent pas effectivement problème, mais cela est pris en compte au niveau du MILP (partie qui n'a pas été développée)

### L'objectif d'équité n'est pas au standard

Le minimax n'est pas un idéal d'équité, et il faut préciser que les décideurs fournissent le même nombre de paires de comparaisons (sinon c'est pas juste, en effet).

## Expérimentations conduites

### Pas d'expériences en cas réel

Contextualiser plus les expériences, que les rôles des décideurs soit plus clair, et présenter un exemple illustratif

### Problème de mise à l'échelle

Oui, c'est un problème, traité dans une autre partie...

### Expériences simples et dans un contexte favorable

Il faut tester sur des cas moins favorables, comme une impossibilité d'avoir un modèle collectif.
Il faut aussi tester avec des groupes plus divers.
Par ailleurs, l'acceptation de révisions est modélisée dans l'espace des modèles (ce qui corrobore le processus utilisant l'espace des modèles)
Dire qu'on cherche à valider le processus, pas le modèle

### Les performances du solveur prennent une trop gande place dans l'analyse des résultats du processus (ça ne démontre que le problème de la mise à l'échelle), il faut d'autres métriques pour évaluer le processus itératif

Donner des informations sur la qualité du compromis, telles que :
- Nombre de refus
- Nombre de changements par itérations
- Nombre de concessions totales
- Répartition de l'effort entre les décideurs

### L'interprétation des résultats sont trop limité pour soutenir les conclusions

Seul avantage des chemins est le nombre réduit d'itérations, pas forcément nombre réduit de modifications. L'intérêt des chemins de préférences doit être plus

### Les expériences conduites ne sont pas très réalistes

Trop d'alternatives (1000) pour trop peu de comparaisons (50 max), il faut réduire le nombre de comparaisons.

<!---
> its originality and contribution to the theory and methodology of multicriteria group decision-making are somewhat limited.

État de l'art plus fourni, en particulier sur la problématique d'éliciter un modèle de préférence collectif (et non juste une recommendation).

> the multicriteria group decision-making ranking problem is presented in a confusing way and not systematized within the fundamental theory of group decision-making; i.e., the problem addressed is neither adequately justified nor clearly presented.

L'état de l'art plus complet devrait résoudre ce problème

> the document does not include an illustrative example that clearly demonstrates how to use the proposed method

À rajouter

> the experiments described in the last section lack a robust experimental design.

Pas tout compris, il faut faire des expériences avec plus de combinaisons de paramètres ? (nombre de critères, nombre de profils, nombre d'exemples, distance entre les décideurs)

---

> I do not find the research motivation and methodological contribution sufficiently clear

État de l'art plus fourni

> the manuscript does not convincingly explain why the proposed framework is needed compared with existing consensus-reaching and preference-adjustment approaches.

Techniquement, c'est pas le même résultat à la fin

> the experimental study is relatively simple and conducted mainly under favorable synthetic settings

Il faut tester sur des cas moins favorables, comme une impossibilité d'avoir un modèle collectif

> The research motivation is not sufficiently clear. The manuscript discusses existing preference elicitation and consensus-reaching approaches, but the specific methodological gap that motivates the proposed framework is not convincingly established.

Gap :

> The novelty of the proposed method appears limited. The collective elicitation model is largely based on an existing MILP, while the preference-path procedure relies on a Greedy Best-First Search. The manuscript should clarify what fundamentally new methodological contribution is provided.

Contribution significatives :

> The claimed advantage of preference paths is not sufficiently justified. Compared with existing minimum-adjustment or consensus-reaching mechanisms, it remains unclear why constructing such paths provides a substantively better decision-aiding process.

À comparer à d'autres processus GDM (CRP, etc.)

> The experimental design is too simple to support the methodological claims. The experiments are entirely based on artificially generated instances with a limited number of criteria, DMs, and reference profiles.

Design trop simple, ou du moins par rapport aux buts affichés ?
Être plus clair sur les objectifs des expérimentations.

> The experimental setting is also relatively favorable to the proposed method, since the instances are generated so that a collectively acceptable SRMP model exists. This assumption substantially simplifies the consensus problem and limits the practical relevance of the results.

Tester des situations sans modèle collectif possible (ainsi que d'autres situations)

> The benchmark used to demonstrate the benefit of preference paths is weak. The main comparison is against randomly ordering the required preference changes, rather than against representative consensus-reaching or preference-adjustment approaches.

À comparer à d'autres processus GDM (CRP, etc.), et il faut que les résultats soit plus probants

> A real-world application or a more realistic decision-making experiment is needed. Without such validation, it is difficult to assess whether the proposed preference paths are understandable and useful for actual decision-makers.

Contextualiser plus les expériences, que les rôles des décideurs soit plus clair

> The computational scalability of the approach is also a concern. The numerical results already indicate rapidly increasing computation time for larger instances, while the manuscript itself acknowledges difficulties in handling large-scale problems.

Oui, c'est un problème, traité dans une autre partie...

---

> The methodological contribution is too limited

contributions vues :

- fonction objectif du SRMP en minimax
- prend en compte les refus des décideurs entre les itérations
- ordonne les révisions par une recherche dans l'espace des modèles
Il manque une validation en cas réel pour que la contribution soit suffisante

> The central "preference path" concept is not adequately motivated at the level at which the DM actually interacts with the method.

justifier le lien entre model-space et preference-space (et pourquoi on se met dans le model-space)

> Most importantly, the paper does not provide convincing evidence that preference paths actually help DMs reach compromise.

À défaut d'avoir des expériences avec des cas réels, présenter au moins un exemple illustratif.

> The simulation used to validate the preference paths is structurally favorable to the proposed mechanism.

Que ce soit via l'existence d'un modèle collectif, mais aussi par le fait que l'acceptation de révisions est modélisée dans l'espace des modèles (ce qui corrobore le processus utilisant l'espace des modèles)

> The experimental DMs are generated as very similar, and the existence of common ground is guaranteed by construction.

Nécessite des expérimentations avec des groupes plus divers.

> The choice of the starting individual model M_i is conceptually problematic.

À partir des L_i, oui il existe plusieurs M_i compatible, et on ne trouvera pas forcément le bon M_i, du moins au début (cela se peaufinera au travers des retours des décideurs). Cependant, il faut réduire le nombre d'alternatives impliquées, car sinon l'ensemble des M_i est trop vaste et perds l'intérêt des chemins (trop facile à trouver)

> GBFS ne propose pas des chemins courts, et le pruning peut créer des gros changements de paramètres

Les affirmations comme quoi les chemins de préférences doivent être courts et proposer de petits chamngements en terme de comparaisons ne sont pas rezspecter par l'utilisation d'un GBFS ainsi que le pruning. Expliquer pourquoi on utilise un GBFS et pas un Astar à l'aide d'expériences ciblées, et préciser que le pruning évite de changer 2 fois la même comparaison.

> Le cas où un modèle n'est pas trouvé n'est pas décrit

Expliciter plus en profondeur ce cas-là

> L'accumulation de refus peut mener à un problème infaisable, et si une étape de chemin est refusée, toutes les comparaisons ne sont pas forcéments refusées.

Être plus clair sur le traitement du cas d'infaisabilité, ainsi que sur la prise en compte des refus par le MILP.

> The claimed fairness interpretation of the minimax objective is stronger than what the standard ensures.

le minimax n'est pas un idéal d'équité, et il faut préciser que les décideurs fournissent le même nombre de paires de comparaisons (sinon c'est pas juste, en effet).

> Les performances du solveur prennent une trop gande place dans l'analyse des résultats du processus (ça ne démontre que le problème de la mise à l'échelle), il faut d'autres métriques pour évaluer le processus itératif

Donner des informations sur la qualité du compromis (tel que *the number of rejected proposals, the number of changed comparisons per proposal, total concessions, and the distribution of burden across DMs.*)

> The interpretation of the experimental results is too limited to support the practical conclusions.

Seul avantage des chemins est le nombre réduit d'itérations, pas forcément nombre réduit de modifications. En général, les analyses des graphes sont limitées.

> The realism of the experimental instances should be discussed.

Trop d'alternatives pour trop peu de comparaisons
