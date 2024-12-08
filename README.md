# TP - PointNet pour la classification de nuages de points

## Objectif

Ce TP permet de reconnaître des formes géométriques définies par des nuages de points. L'objectif est d'implémenter une architecture de réseau neuronal comprenant les réseaux `TNet` et `PointNet` pour la classification des formes géométriques à partir de nuages de points.

---

## Prérequis

### Librairies et logiciels requis

La programmation est réalisée en Python et nécessite les librairies suivantes :

- **Numpy** : pour la manipulation de données numériques.
- **Matplotlib** : pour visualiser l’évolution des fonctions de coût.
- **PyTorch** : pour la construction et l'entraînement des réseaux neuronaux.
- **Scikit-learn** : pour la matrice de confusion (si besoin).
  
Un environnement Conda `tp_ml.yml` est fourni sur la page du cours pour la gestion des dépendances.

---

## Données

Un script `prepare_data.py` est fourni et génère des données de test et d'entraînement sous forme de nuages de points. Ces données sont divisées en deux ensembles : **train** et **test**. Les labels des données sont définis comme suit :

- **Label 0** : Cylindres
- **Label 1** : Parallélépipèdes rectangles
- **Label 2** : Tores

Les données peuvent être visualisées à l'aide de **Matplotlib**.

---

## Architecture du modèle

### 1. TNet (Transformation Network)

Le **TNet** est utilisé pour générer une matrice de rotation afin d'aligner les données d'entrée.

**Structure** :
- **Entrée** : Nuage de points nD sous la forme `batch_size x nD x 2048`
- **Trois couches de convolution 1D** avec des tailles respectives de 64, 128 et 1024, suivies de **BatchNorm** et **ReLU**.
- **MaxPooling** par canal avec `torch.max`.
- **Deux couches linéaires** avec des tailles 512 et 256, suivies de **BatchNorm** et **ReLU**.
- **Une couche linéaire** passant de 256 à `nD * nD`.

### 2. PointNet

Le modèle **PointNet** utilise le **TNet** pour aligner les données et les caractéristiques extraites.

**Structure** :
- **TNet avec nD=3** : pour aligner les données.
- Application de la matrice de transformation aux données d'entrée avec `torch.bmm`.
- **Convolution 1D (64)** suivie de **BatchNorm** et **ReLU**.
- **TNet avec nD=64** : pour aligner les caractéristiques extraites.
- **Convolution 1D (128)** suivie de **BatchNorm** et **ReLU**.
- **Convolution 1D (1024)** suivie de **BatchNorm** et **ReLU**.
- **MaxPooling** par canal avec `torch.max`.
- **Deux couches linéaires** avec des tailles 512 et 256, suivies de **BatchNorm** et **ReLU**.
- **Une couche linéaire** passant de 256 à 3 (pour les 3 classes) avec **log_softmax**.

### 3. Fonction de perte et optimisation

- La fonction de perte utilisée est la **Negative Log Likelihood** (`nll_loss`).
- L'optimisation se fait via **Stochastic Gradient Descent (SGD)** avec un taux d'apprentissage de **1e-3**.

---

## Test et évaluation

### 1. Tester sur les données de test

Après l'entraînement, le modèle peut être testé sur l'ensemble de données de test. Les résultats peuvent être évalués en calculant :

- Les taux de **vrais/faux positifs** et **vrais/faux négatifs**.
- Le taux de **bonne reconnaissance**.
- La **matrice de confusion** via la bibliothèque **scikit-learn**.

---

## Robustesse au bruit

L'algorithme peut être testé pour sa robustesse face au bruit. Il est recommandé d'ajouter du bruit aux données et de tester les performances du modèle. Une autre approche consiste à réentraîner le modèle en augmentant les données avec du bruit aléatoire pour tester sa résilience.

---

## Instructions d'exécution

1. Clonez ce repository sur votre machine.
2. Créez un environnement Conda à partir du fichier `tp_ml.yml` fourni.
3. Exécutez le script `prepare_data.py` pour générer les données de test et d'entraînement.
4. Lancez l'entraînement avec le script principal et surveillez l'évolution des fonctions de coût.
5. Testez le modèle avec les données de test et évaluez les résultats.

---
