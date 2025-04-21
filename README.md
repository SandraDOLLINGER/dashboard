# Dashboard de Scoring Crédit – Prêts à Dépenser
Ce dashboard interactif es destiné aux chargés de relation client de la société **Prêts à dépenser**.
Cette société propose des crédits à la consommation à des clients ayant peu ou pas d'historique de prêt.
Il permet aux chargés de clientèle d’expliquer de manière transparente les décisions d’octroi ou de refus de crédit.

## Cahier des charges
Le dashboard répond aux spécifications suivantes :
- Visualiser le **score de crédit**, sa **probabilité** (et sa proximité avec le **seuil de décision**) de manière intelligible pour des non-experts.
- Accéder aux **principales informations descriptives du client**.
- **Comparer le profil d’un client** à l’ensemble des clients ou à un **groupe similaire** (via des filtres interactifs).
- Intégrer des critères d’**accessibilité** conformes aux **normes WCAG** pour les personnes en situation de handicap (notamment dans les graphiques).
- Déployer l’outil sur le **Cloud** pour un accès facilité depuis n’importe quel poste.

---

Ce dashboard a été developpé avec Streamlit et appelle l'API de prédiction déployée sur [Heroku](https://application-prediction-scoring-b81541cc2c3b.herokuapp.com/predict/) et dont le code est disponible sur le [repository suivant](https://github.com/SandraDOLLINGER/modelisation_scoring).

---

## Fonctionnalités
Après avoir sélectionné un client sur la page d'accueil, 3 onglets sont disponibles :
### 1. Prédiction
- Probabilité d’un **incident de paiement**.
- **Décision finale** concernant le prêt (accordé ou refusé).
- Principales **informations personnelles** du client.
- Une **jauge colorée** permet de visualiser la distance par rapport au **seuil de décision**.
### 2. Explicabilité
- **Explication globale** : variables influentes au niveau du modèle.
- **Explication locale** : variables ayant impacté la décision pour ce client.
### 3. Analyse
Des graphiques permettent de situer le client dans l’ensemble des données, en distinguant les clients avec ou sans incident :
- Histogramme
- Boxplot
- Analyse bivariée
- Densité estimée (KDE)

---


## Installation locale
Cloner le dépot, créer un environnement virtuel, installer les dépendances :

    git clone https://github.com/SandraDOLLINGER/dashboard
    cd dashboard
    python -m venv venv
    source venv/bin/activate  ou venv\Scripts\activate sur Windows
    pip install -r requirements.txt

## Déploiement sur Heroku
Déploiement manuel actuellement.
