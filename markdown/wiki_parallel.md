# Paralléliser un code Python

Une des manières la plus simple et rapide à mettre en place pour diminuer le temps d'exécution d'un code est de le paralléliser, c'est à dire de l'exécuter sur plusieurs coeurs (ou CPUs). Le gain en temps va dépendre directement du nombres de coeurs utilisés, donc cette solution est plus efficace quand elle est employée sur un serveur de calcul avec plusieurs coeurs à disposition. Cependant, pour pouvoir être parallélisé, le code doit être divisé en plusieurs tâches indépendantes entres elles.

## Contexte

Imaginons que l'on veuille tester différentes combinaisons de paramètres pour une méthode dans le but de trouver celle qui donne les meilleurs résultats.

Soit une méthode d'apprentissage $f$ qui dépend de 3 paramètres $\alpha \in [0, 1]$, $\beta \in [0, 1]$ et $\gamma \in [0, 1]$ et qui, appliquée sur une donnée $x = (x_1, x_2)$, renvoie une prédiction $f(x) = \alpha x_1 + \beta x_2 + \gamma$ :

```python
def f(x, alpha, beta, gamma):
    return alpha * x[0] + beta * x[1] + gamma
```

Soit un ensemble de données $\mathcal{D}$ qui contient des données $x$ ainsi que leurs observations $y$.

```python
D = [
    ([1, 2], 3),
    ([4, 5], 6),
    ([7, 8], 9),
]
```

On cherche à déterminer les paramètres $\alpha$, $\beta$ et $\gamma$ qui minimisent la somme des carrés des erreurs de prédiction : $S = \sum_{(x, y) \in \mathcal{D}} (y - f(x))^2$. On teste toutes les combinaisons de paramètres pour lesquelles les paramètres prennent leurs valeurs dans $[0, 0.25, 0.5, 0.75, 1]$, ce qui fait donc $5^3 = 125$ combinaisons à tester. Le code python suivant génère toutes les combinaisons de paramètres, calcule l'erreur $S$ de la méthode $f$ sur l'ensemble de données d'entrainement $\mathcal{D}$ avec ces paramètres, et choisit la combinaison de paramètres minimisant l'erreur.

```python
results = {} # dictionnaire qui à chaque combinaison de paramètres associe l'erreur calculée sur l'ensemble de données d'entrainement

for alpha in [0, 0.25, 0.5, 0.75, 1]:
    for beta in [0, 0.25, 0.5, 0.75, 1]:
        for gamma in [0, 0.25, 0.5, 0.75, 1]:
            combinaison = (alpha, beta, gamma)

            error = 0 # initialise l'erreur calculée

            for (x, y) in D: # pour chaque donnée d'entrainement
                error += (y - f(x, alpha, beta, gamma))**2 # calcule l'erreur

            results[combinaison] = error # enregistre l'erreur associée à la combinaison de paramètres testée

best_combinaison = min(results, key=results.get) # sélectionne la combinaison de paramètres avec l'erreur minimale
```

Ce code trouve bien la meilleure combinaison de paramètres, cependant il teste l'une après l'autre chaque combinaison. Ce qui fait que si, par exemple, l'éxécution de la méthode $f$ dure une minute, alors tout le processus mettra deux heures. Pour réduire ce temps de calcul, il faut alors pouvoir tester plusieurs combinaisons de paramètres en même temps.

## Parallélisation du code

Tout d'abord, il faut isoler le code qui s'applique à une combinaison de paramètres définie, et l'encapsuler dans une nouvelle fonction (`compute_error` ici). Il ne faut pas oublier d'intégrer les variables utilisées dans la fonction et définies en dehors de celle-ci en tant qu'arguments (`D` dans notre exemple). Il faut veiller à éviter au maximum de modifier des variables définies en dehors de la fonction, pour que chaque appel à cette fonction soit totalement indépendant des autres. Ce code calcul l'erreur de la méthode $f$ sur l'ensemble de données $\mathcal{D}$, et l'enregistre au côté de la combinaison testée.

```python
def compute_error(alpha, beta, gamma, D)
    combinaison = (alpha, beta, gamma)

    error = 0 # initialise l'erreur calculée

    for (x, y) in D: # pour chaque donnée d'entrainement
        error += (y - f(x, alpha, beta, gamma))**2 # calcule l'erreur

    return error
```

Ensuite, il faut transformer cette fonction pour qu'elle ne prenne en entrée que les paramètres que l'on veut tester. Pour cela, nous avons besoin d'utiliser la fonction `partial` du module `functools` de la librairie standard, qui "fixe" les arguments d'une fonction.

```python
from functools import partial

compute_error_from_combinaison = partial(compute_error, D=D)
```

Maintenant que le code qui sera exécuté pour chaque combinaison est défini, il nous faut lister les combinaisons qui seront testées.

```python
```