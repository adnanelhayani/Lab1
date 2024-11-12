# Synthèse des Apprentissages

## Première Partie : Prévision des Prix avec Réseaux de Neurones et Techniques de Régularisation

### 1. Explorer et Préparer un Jeu de Données
- J'ai appris à importer et nettoyer un jeu de données avec `pandas`, incluant la gestion des valeurs manquantes, la suppression des doublons et la conversion des variables catégorielles en valeurs numériques.
- En utilisant `seaborn` et `matplotlib`, j'ai appliqué des techniques de visualisation pour mieux comprendre les relations entre les variables et détecter des anomalies dans le jeu de données.

### 2. Construire un Modèle de Régression avec PyTorch
- J'ai défini une architecture de réseau neuronal profond (DNN) pour une tâche de régression en utilisant PyTorch, en créant un modèle avec des couches `Linear` et des fonctions d'activation `ReLU` pour prédire des valeurs continues.
- J'ai exploré l'optimisation du modèle en utilisant l'optimiseur Adam, en suivant l'évolution de la perte pendant l'entraînement.

### 3. Appliquer des Techniques de Régularisation pour Améliorer la Performance du Modèle
- J'ai implémenté les techniques de régularisation Dropout et L2 Regularization pour éviter le sur-apprentissage (overfitting).
- En ajoutant Dropout et en utilisant un `weight_decay` avec l'optimiseur Adam, j'ai observé une meilleure généralisation sur les données de test.

### 4. Suivi de la Performance du Modèle
- J'ai appris à suivre les performances du modèle en traçant les courbes de perte sur les données d'entraînement et de test sur plusieurs époques pour visualiser l'impact de la régularisation sur la réduction du sur-apprentissage.

---

## Deuxième Partie : Classification de Données pour la Maintenance Prédictive

### 1. Préparation et Nettoyage des Données
- Chargement et préparation d'un dataset de maintenance prédictive, comprenant la gestion des valeurs manquantes, l'encodage des variables catégorielles et la normalisation des caractéristiques.
- Pour traiter le déséquilibre des classes, une technique de suréchantillonnage a été appliquée.

### 2. Construction du Modèle de Classification
- Nous avons conçu un réseau neuronal à trois couches en utilisant PyTorch et optimisé le modèle avec Adam, en suivant la perte et la précision pendant l'entraînement.

### 3. Application des Techniques de Régularisation
- Afin de réduire le surapprentissage, plusieurs techniques de régularisation ont été appliquées :
  - **Dropout** : pour désactiver aléatoirement certains neurones pendant l'entraînement.
  - **Décroissance de poids (L2)** : régularisation pour pénaliser les poids trop élevés.
  - **Normalisation par lot (Batch Normalization)** : pour normaliser les entrées des couches et accélérer l'entraînement.

### 4. Évaluation du Modèle
- Le modèle a été évalué avec des métriques telles que l'exactitude (accuracy), le score F1, la sensibilité (recall), et la précision (precision) pour comparer les performances avec et sans régularisation.

### 5. Comparaison des Résultats
- La régularisation a permis de réduire le surapprentissage et d'améliorer la performance sur les données de test par rapport au modèle sans régularisation.

### 6. Conclusion et Perspectives
- Ce laboratoire a permis de mieux comprendre l'impact de la régularisation et des techniques de normalisation sur les réseaux neuronaux.
- Manipulation de bibliothèques de machine learning telles que Scikit-learn pour le prétraitement des données et PyTorch pour la création et l'entraînement des modèles de deep learning.
