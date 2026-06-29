# Paralléliser un code Python

Une des manières la plus simple et rapide à mettre en place pour diminuer le temps d'exécution d'un code est de le paralléliser, c'est à dire de l'exécuter sur plusieurs coeurs (ou CPUs). Le gain en temps va dépendre directement du nombres de coeurs utilisés, donc cette solution est plus efficace quand elle est employée sur un serveur de calcul avec plusieurs coeurs à disposition. Cependant, pour pouvoir être parallélisé, le code doit être divisé en plusieurs tâches indépendantes entres elles.

## Exemple

Imaginons que l'on veuille tester différentes combinaisons de paramètres pour une méthode dans le but de trouver celle qui donne les meilleurs résultats.

Soit une méthode d'apprentissage `dummy`, qui dépend de 3 paramètres $\alpha$, $\beta$ et $\gamma$, et qui, appliquée sur un ensemble de données d'entraînement, renvoie un taux d'erreur à minimiser :

```python
def dummy(alpha, beta, gamma, train):
    error = 0

    for data in train:
        prediction = alpha * data[0] + beta * data[1] + gamma * data[2]
        target = data[3]
        error += abs(target - prediction)

    return error
```
