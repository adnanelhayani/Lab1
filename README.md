# Rapport de Projet - Prévision des Prix avec Réseaux de Neurones et Techniques de Régularisation

## Objectifs du partie 1

Ce projet vise à prédire les prix de clôture d'un actif financier en utilisant un modèle de réseau de neurones profond (DNN). Le modèle a été entraîné avec et sans techniques de régularisation pour comparer les performances des deux approches.

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

**Note** : Ce projet utilise la bibliothèque PyTorch pour la modélisation et le calcul des métriques de performance. Les détails des hyperparamètres et de l'architecture sont dans le code associé.
