# Rapport de Projet - Prévision des Prix avec Réseaux de Neurones et Techniques de Régularisation

## Objectifs du partie 1

Cette partie vise à prédire les prix de clôture d'un actif financier en utilisant un modèle de réseau de neurones profond (DNN). Le modèle a été entraîné avec et sans techniques de régularisation pour comparer les performances des deux approches.

## Étapes du partie 1

### 1. Préparation des Données
Les données financières ont été extraites d'un fichier CSV contenant des informations sur les prix et le volume de transactions. Les étapes de préparation ont inclus :
- Transformation de la colonne `symbol` en valeurs numériques
- Suppression de la colonne `date`, jugée non pertinente
- Suppression des lignes dupliquées et vides (NaN)
- Normalisation des colonnes pertinentes (`open`, `close`, `low`, `high`, `volume`)

### 2. Visualisation des Données
Des graphiques ont été créés pour explorer les relations entre les variables et mieux comprendre les données :
- Pairplot : relations entre les variables de prix et le volume
- Matrice de corrélation : identification des relations entre les variables
- Courbes de prix au fil du temps
- Histogramme : distribution du volume des transactions

### 3. Modélisation avec Réseau de Neurones (DNN)
Un réseau de neurones avec trois couches cachées a été implémenté en utilisant PyTorch pour prédire les prix de clôture :
- Division des données en ensembles d'entraînement et de test
- Normalisation des données d'entrée pour améliorer la convergence
- Entraînement sur 100 époques avec l'optimiseur Adam et la fonction de perte MSE (Erreur Quadratique Moyenne)

### 4. Optimisation d'Hyperparamètres
Un GridSearch a été effectué avec MLPRegressor pour trouver les meilleurs paramètres du modèle, notamment la taille des couches cachées et les fonctions d'activation.

### 5. Modélisation avec Régularisation
Une version améliorée du modèle a été développée en intégrant :
- Dropout pour éviter le sur-apprentissage
- Régularisation L2 via le paramètre `weight_decay` de l'optimiseur Adam
- Comparaison des performances entre le modèle de base et le modèle régularisé pour évaluer l'impact de la régularisation

### 6. Évaluation du Modèle
Les modèles ont été évalués en utilisant :
- MSE (Mean Squared Error) comme mesure de la perte de test
- Coefficient de détermination R² pour évaluer la qualité des prédictions
- Courbes de perte et de R² tracées pour visualiser l'évolution au fil des époques

### 7. Comparaison des Résultats
Les performances du modèle avec et sans régularisation ont été comparées :
- Le modèle régularisé a montré une meilleure généralisation sur les données de test
- Le modèle sans régularisation tendait à sur-apprendre les données d'entraînement

## Conclusions
- Les techniques de régularisation (Dropout, régularisation L2) ont permis de réduire le sur-apprentissage et d'améliorer la performance du modèle.
- Les courbes de perte et R² ont montré une amélioration de l'évaluation du modèle régularisé.
- Le modèle sans régularisation présentait un sur-apprentissage, tandis que le modèle régularisé a mieux généralisé sur les données de test.

---

# Partie 2 de Classification pour la Maintenance Prédictive

Cette partie de classification utilise un modèle de réseau de neurones profonds pour prédire les types de défaillances potentielles dans un processus de maintenance prédictive. Voici les étapes suivies dans ce projet, de la préparation des données à l'entraînement du modèle, ainsi que les résultats des métriques de performance.

## Contenu
1. [Prétraitement des Données](#prétraitement-des-données)
2. [Exploration des Données](#exploration-des-données)
3. [Équilibrage des Classes](#équilibrage-des-classes)
4. [Entraînement du Modèle de Classification](#entraînement-du-modèle-de-classification)
5. [Optimisation des Hyperparamètres](#optimisation-des-hyperparamètres)
6. [Régularisation et Amélioration du Modèle](#régularisation-et-amélioration-du-modèle)
7. [Résultats](#résultats)
8. [Conclusion](#conclusion)

## Prétraitement des Données

1. **Chargement des Données** : Chargement du dataset `predictive_maintenance.csv` et suppression des colonnes non pertinentes (`UDI` et `Product ID`).
2. **Encodage des Variables Catégorielles** : Conversion des valeurs textuelles en valeurs numériques pour les colonnes `Type` et `Failure Type` en utilisant `LabelEncoder`.
3. **Normalisation des Données** : Utilisation de `StandardScaler` pour normaliser les caractéristiques sélectionnées (`Air temperature`, `Process temperature`, etc.).

## Exploration des Données

1. Création d'histogrammes pour visualiser les distributions des caractéristiques.
2. Affichage des matrices de corrélation pour comprendre les relations entre les caractéristiques.
3. Visualisation de la distribution des caractéristiques après normalisation.

## Équilibrage des Classes

Pour améliorer l'efficacité du modèle, un rééchantillonnage de la classe minoritaire (`Target=1`) a été effectué afin d'équilibrer le dataset.

## Entraînement du Modèle de Classification

Un modèle de réseau de neurones profonds a été défini à l'aide de PyTorch, avec les étapes suivantes :

1. **Définition du Modèle** : Création d'un modèle `DeepNN` avec deux couches cachées et une fonction d'activation ReLU.
2. **Entraînement** : Entraînement du modèle sur 50 époques en utilisant `CrossEntropyLoss` comme fonction de perte et `Adam` comme optimiseur.
3. **Évaluation** : Calcul de la précision sur l'ensemble de test.

## Optimisation des Hyperparamètres

Un `GridSearchCV` a été appliqué avec `skorch` pour trouver les meilleurs hyperparamètres (learning rate, nombre d'époques, optimiseur) et ainsi améliorer les performances du modèle.

## Régularisation et Amélioration du Modèle

Afin de réduire le surapprentissage, un modèle avec régularisation a été implémenté :

1. **Batch Normalization et Dropout** : Ajoutés après chaque couche cachée pour stabiliser l'entraînement et prévenir l'overfitting.
2. **L2 Regularization** : Utilisation de `weight_decay` dans l'optimiseur `Adam` pour renforcer la généralisation du modèle.

## Résultats

### Modèle sans Régularisation
- **Précision** : 98,80%
- **F1 Score** : 98,74%
- **Recall** : 98,80%
- **Précision** : 98,73%

### Modèle avec Régularisation
- **Précision** : 99,60%
- **F1 Score** : 99,61%
- **Recall** : 99,60%
- **Précision** : 99,62%

Les résultats montrent une amélioration significative après régularisation.

## Conclusion

Ce projet a permis de se familiariser avec les étapes clés de la création d'un modèle de classification en deep learning, en particulier :

- Le prétraitement et la normalisation des données,
- L'équilibrage des classes pour une meilleure précision,
- L'optimisation des hyperparamètres pour maximiser la performance,
- L'importance des techniques de régularisation pour un modèle plus général.

Ce code et ses résultats sont disponibles dans ce répertoire pour référence et utilisation future.

