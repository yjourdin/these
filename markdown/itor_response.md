# Author response

We would like to thanks the editor and the referees for their useful comments. The additions and deletions compared to the previous version of the manuscript are respectively highlighted in blue and red in the additional file for review.

## Referee 1

> In page 7, in Algorithm 1, in line 6, as far as I understand, the problem is a maximization problem, according to simulated annealing pseudo code, if the neighbour solution is better (i.e. has bigger objective function value) than the current solution, the neighbor solution should be accepted as the new current solution in default, but if the neighbour solution is worse (i.e. has lower objective function value) than the current solution, the neighbor solution should be accepted as the new current solution with a probability compared to a generated random number. In line 6, only the worse case is written; the better case is not. Please check the pseudo code.

The line 6 in question : $\exp((f(\mathcal{M}', \mathcal{D}_{train}) - f(\mathcal{M}, \mathcal{D}_{train})) / T) > random(0, 1)$

In the worse case, $f(\mathcal{M}', \mathcal{D}_{train}) < f(\mathcal{M}, \mathcal{D}_{train})$, thus $f(\mathcal{M}', \mathcal{D}_{train}) - f(\mathcal{M}, \mathcal{D}_{train}) < 0$ and then $P := \exp((f(\mathcal{M}', \mathcal{D}_{train}) - f(\mathcal{M}, \mathcal{D}_{train})) / T) < 1$
The solution is accepted if $P > random(0, 1)$, thus with a probability equals to $P < 1$.

In the better case, $f(\mathcal{M}', \mathcal{D}_{train}) > f(\mathcal{M}, \mathcal{D}_{train})$, thus $f(\mathcal{M}', \mathcal{D}_{train}) - f(\mathcal{M}, \mathcal{D}_{train}) > 0$ and then $P := \exp((f(\mathcal{M}', \mathcal{D}_{train}) - f(\mathcal{M}, \mathcal{D}_{train})) / T) > 1$.
So, since $P > 1$, it will be greater than any generated random number between 0 and 1, so it will be always accepted, as expected in the better case.

---

> The authors wrote that in the literature, there is at least one mathematical model and a genetic algorithm that solves the same problem. Since the simulated annealing algorithm does not guarantee the optimal, I wonder if to see the performance of the proposed simulated annealing algorithm better, is there a chance to make a comparison with these other methods?

Il faut comparer les résultats présentés dans l'article à ceux des autres méthodes présentées dans l'état de l'art.
Sur la forme, juste un paragraphe dans la section Résultats à la fin qui aborde ce sujet là, ou un tableau récapitulatif (comme dans (Khannoussi et al., 2024)), ou directement sur les graphs ?

## Referee 2

> For the abstract, the authors should clearly justify the main contribution of the paper instead of mentioning too much about the background.

Multiple Criteria Decision Aiding (MCDA) helps decision-makers (DMs) to reach better decisions in multi-criteria problems using preference models, whose parameters are elicited to best correspond to the preferences of the DM using, among other things, holistic judgments.
Regarding the elicitation of the Reference based on Multiple reference Profiles (RMP) model, the literature only contains an exact method based on a Boolean Satisfiability formulation, while a mixed-integer linear program and evolutionary metaheuristics focus on a more simpler version of the model (SRMP).
Exact methods for preference elicitation usually struggle to solve cases with many criteria and a lot of preference information, while metaheuristics have a gap to optimal solutions.
To address these two issues, we propose in this article a simulated annealing based metaheuristic to elicit an RMP model.
In order to evaluate the performances of this method, we conducted numerical experiments on simulated instances randomly and uniformly generated.
The results shows that the proposed method is able to solve at optimality big instances, as well as being closer to optimal solutions than other metaheuristics in SRMP elicitation.

---

> In the introduction section, the authors should clarify the research gaps or limitations of existing studies to motivate this paper.

However, the exact methods in the literature struggle to solve big instances, and their performances have not been assessed on holistic judgements derived from the more general RMP model. Regarding the metaheuristics present in the state of the art, some only focus on the more specific SRMP model, and all have many hyperparameters to tune for this case of preference elicitation.

---

> The paper aims to learn preference models based on indirect preference information. Some related studies can be included to enhance the literature review or make comparisons, for instance, Constructive preference elicitation for multi-criteria decision analysis using an estimate-then-select strategy; Integrating machine learning models to learn potentially non-monotonic preferences for multi-criteria sorting from large-scale assignment examples; An incremental preference elicitation-based approach to learning potentially non-monotonic preferences in multi-criteria sorting.

- "Constructive preference elicitation for multi-criteria decision analysis using an estimate-then-select strategy", Liang, Q., Zhang, Z., & Su, Y. (2025) *Information Fusion* cité 11 fois
  - Élicitation incrémentale sur le modèle additif
- "Integrating machine learning models to learn potentially non-monotonic preferences for multi-criteria sorting from large-scale assignment examples", Li, Z., Zhang, Z., & Pedrycz, W. (2025) *Omega* cité 8 fois
  - Réseaux de neurones appliqués au problème de tri
- "An incremental preference elicitation-based approach to learning potentially non-monotonic preferences in multi-criteria sorting", Li, Z., Zhang, Z., & Pedrycz, W. (2025) *European Journal of Operational Research* cité 7 fois
  - Élicitation incrémentale pour le problème de tri

On les rajoute ou pas ? 3x le même auteur (volonté de promouvoir ses propres travaux récents ?), et pas d'articles venant de ITOR

Mais en effet, je pourrais citer des articles quand je présente l'élicitation des préférences (directe et indirecte)

---

> For the learning model, the authors should clearly show what are the parameters that should be learned. It is inappropriate to use model as the input and output.

In the case of the elicitation of an RMP model, the parameters to be inferred are the performance of the reference profiles $\mathcal{P}$, the lexicographic order $\sigma$ and the importance relation $\trianglerighteq$ (replaced by the weights for an SRMP model).

A paragraph got added in the beginning of section 3 which details the parameters of the model that are inferred by the proposed algorithm.

---

> The notations in Algorithm 1 should be clearly defined. Otherwise, it is difficult to understand the algorithm. Howe to initialize the current and best known model? It seems these points are not clearly mentioned in the paper.

The different initialisation steps have been integrated in the algorithm for more transparency, while the last paragraph of section 3 was already explaining them. Moreover, the input data of the algorithm got explained.

Techniquement, toutes les notations sont définies, mais certaines le sont 2 pages plus loin, donc je pourrais les définir plus tôt, et en parler 2 pages plus loin comme actuellement.
La génération de $\mathcal{M}_{init}$ (initial model) est détaillée dans la partie 4.2 (c'est précisé à la fin de la partie 3 sur l'algorithme).

---

> I suggest the authors use a simple example to show the implementation of the algorithm to enhance the readability of the paper.

A visual example for every move introduced in our paper got added, for more clarity.

---

> Why SA instead of other metaheuristic should be used in the paper?

The simulated annealing metaheuristic, apart from being popular to solve several optimization problems, got chosen for its simplicity compared to other metaheuristics. Indeed, only move operators must be defined, as well as two hyperparameters (cooling ratio $\alpha$ and initial acceptance rate $\chi_{init}$), compared to the crossover and mutation operators and several other hyperparameters from the evolutionary metaheuristics.

---

> It would be better for the authors to provide some comparative studies with existing preference learning model based on pairwise comparisons over alternatives.

Même remarque que le reviewer précédent.