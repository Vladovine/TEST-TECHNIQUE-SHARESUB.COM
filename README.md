# TEST TECHNIQUE SHARESUB
# Risk Monitor - Cas pratique - Chief of Staff Direction des Opérations

Cas pratique technique pour la Direction des Opérations de Sharesub.

Ce projet vise à concevoir un outil interne capable de :

- comprendre et nettoyer une base SQLite volontairement dégradée ;
- produire un score de risque reproductible par subscriber ;
- fournir une interface opérationnelle lisible ;
- intégrer un agent IA pour aider à l’analyse et à la décision ;
- documenter clairement les choix, limites et compromis.

---

## Livrables

1. **Repo GitHub** — code structuré, commits atomiques et progression visible dans l’historique.
2. **Notebook exploratoire** — démarche de data cleaning, pas seulement résultat final.
3. **App Streamlit** — exécutable en local.
4. **README.md** — hypothèses sur les données, choix techniques argumentés, architecture, limites connues, pistes d’évolution.
5. **Prompts IA versionnés** — dans `prompts/`, avec explication de chaque itération.

---

## Arborescence

```text
risk-monitor/
├─ README.md
├─ app.py
├─ src/
│  ├─ __init__.py
│  ├─ scoring.py
│  └─ agent.py
├─ notebooks/
│  ├─ 01_exploration.ipynb
│  ├─ 02_cleaning_and_normalisation.ipynb
│  └─ 03_features_and_scoring.ipynb
├─ prompts/
│  ├─ agent_analyst_v1.txt
│  ├─ agent_decider_v1.txt
│  └─ README.md
├─ data/
│  ├─ raw/
│  │  └─ risk_monitor_dataset.sqlite
│  ├─ processed/
│  │  └─ risk_monitor_clean.sqlite
│  ├─ app_state.sqlite
│  └─ agent_logs/
├─ outputs/
│  └─ subscribers_risk_scored.csv
└─ .gitignore
```

---

## Hypothèses sur les données

Le jeu de données a été fourni avec des anomalies volontairement introduites :

- dates hétérogènes ;
- statuts numériques non documentés ;
- doublons partiels ;
- lignes orphelines ;
- valeurs manquantes ;
- quelques dates d’inscription incohérentes dans le futur.

Les choix retenus sont les suivants :

- la base brute n’est jamais modifiée ;
- le nettoyage est fait dans une base SQLite séparée ;
- les colonnes d’origine sont conservées quand elles apportent de la traçabilité ;
- les dates sont normalisées en UTC ;
- les doublons stricts de `payments` sont supprimés ;
- les valeurs manquantes sont conservées dans les données et simplement rendues lisibles à l’écran ;
- la date de référence du scoring est calculée à partir des signaux opérationnels uniquement.

Les dates d’inscription anormales dans le futur ne sont pas utilisées pour piloter la date de référence du scoring.

---

## Choix techniques

### 1. Exploration avant transformation
Le notebook d’exploration a servi à :

- lire le schéma des tables ;
- mesurer les volumes ;
- observer les premiers enregistrements ;
- repérer les statuts ;
- détecter les doublons ;
- vérifier les relations entre tables ;
- identifier les anomalies à traiter.

### 2. Nettoyage traçable
Le notebook de nettoyage :

- documente les anomalies ;
- applique des règles de nettoyage explicites ;
- conserve les colonnes brutes dans des champs `_raw` ;
- exporte une base SQLite nettoyée.

### 3. Scoring reproductible
Le scoring est déterministe :

- mêmes données = même score ;
- aucune dépendance au hasard ;
- aucune dépendance à un appel LLM.

Le score repose sur quatre familles de signaux :

- paiements ;
- memberships ;
- réclamations ;
- récence d’activité.

### 4. Industrialisation du scoring
La logique de scoring est extraite dans `src/scoring.py` pour pouvoir être relancée en une commande :

```bash
python -m src.scoring --input data/processed/risk_monitor_clean.sqlite --output outputs/subscribers_risk_scored.csv
```

### 5. Interface interne
L’app Streamlit permet :

- de voir les subscribers les plus risqués ;
- de filtrer la vue ;
- d’ouvrir une fiche détaillée ;
- de consulter l’historique ;
- de prendre une action locale ;
- de consulter une analyse IA.

### 6. Agent IA
L’agent IA remplit deux rôles :

- **analyste** : produire un résumé structuré du cas ;
- **décideur** : proposer une recommandation d’action.

Les prompts sont versionnés dans `prompts/` et les appels sont journalisés localement.

---

## Architecture du projet

### Exploration
`notebooks/01_exploration.ipynb` sert à comprendre les données avant toute transformation.

### Nettoyage
`notebooks/02_cleaning_and_normalisation.ipynb` produit la base nettoyée :
`data/processed/risk_monitor_clean.sqlite`

### Scoring
`notebooks/03_features_and_scoring.ipynb` documente la construction du score.  
La version exécutable est `src/scoring.py`.

### Interface
`app.py` lit :

- `outputs/subscribers_risk_scored.csv`
- `data/processed/risk_monitor_clean.sqlite`

et stocke les actions locales dans :

- `data/app_state.sqlite`

### Agent IA
`src/agent.py` s’appuie sur :

- les prompts dans `prompts/`
- les logs dans `data/agent_logs/`

---

## Résultat du scoring

Le score final est exprimé sur 100 et classé en 4 niveaux :

- `low`
- `watch`
- `high`
- `critical`

Il sert à prioriser l’action opérationnelle.  
Il ne doit pas être interprété comme une vérité absolue, mais comme un outil de tri et de surveillance.

---

## Interface Streamlit

L’interface a été conçue pour être utilisable sans documentation.  
La page suit une logique simple :

1. les subscribers prioritaires apparaissent en premier ;
2. la fiche détaillée apparaît en dessous ;
3. l’historique et les actions sont accessibles dans la même zone ;
4. les valeurs manquantes sont affichées de manière lisible.

L’objectif est qu’un opérateur puisse comprendre rapidement :

- qui est le plus à risque ;
- pourquoi il remonte ;
- quelle action prendre.

---

## Gestion des valeurs manquantes

Le jeu de données contient volontairement des valeurs manquantes.  
Elles sont :

- conservées dans les données pour ne pas casser la traçabilité ;
- affichées sous une forme lisible (`—`) dans l’interface ;
- laissées intactes dans les calculs lorsque leur absence a du sens métier.

---

## Limites connues

Le projet reste tributaire de la qualité des données source :

- certaines dates sont incohérentes ;
- certains statuts sont partiellement documentés ;
- certaines lignes sont orphelines ;
- certaines valeurs manquantes sont structurelles ;
- certaines dates d’inscription apparaissent dans le futur.

Ces limites doivent être lues comme des contraintes de la donnée, pas comme des erreurs de conception du projet.

---

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install pandas numpy streamlit openai
```

Si un fichier `requirements.txt` est présent, il peut être utilisé à la place.

---

## Lancement du scoring

```bash
python -m src.scoring --input data/processed/risk_monitor_clean.sqlite --output outputs/subscribers_risk_scored.csv
```

---

## Lancement de l’app

```bash
streamlit run app.py
```

---

## Prompts IA

Les prompts sont versionnés dans `prompts/` pour garder une trace claire des itérations.

- `agent_analyst_v1.txt` : résumé structuré du cas ;
- `agent_decider_v1.txt` : recommandation d’action ;
- `prompts/README.md` : explication des versions.

---

## Fichiers générés localement

Le projet produit plusieurs fichiers qui ne sont pas versionnés :

- `data/processed/`
- `outputs/`
- `data/app_state.sqlite`
- `data/agent_logs/`

Ces fichiers sont générés par le pipeline, l’app Streamlit ou l’agent IA.
Cependant, à la fin du projet, j'ai push le fichier sqlite nettoyé et le CSV Scoré pour que vous y aillez accès.                                             

---

## Ce qui a été livré

- un notebook d’exploration ;
- un notebook de nettoyage ;
- un notebook de scoring ;
- un module de scoring réutilisable ;
- une interface Streamlit ;
- un agent IA ;
- des prompts IA versionnés ;
- une base SQLite nettoyée ;
- un CSV scoré ;
- une documentation complète.

---

## Remarque finale

Le dépôt a été construit avec une progression visible dans l’historique Git :

- exploration ;
- nettoyage ;
- scoring ;
- industrialisation ;
- interface ;
- agent IA ;
- documentation.

Le but n’était pas seulement d’obtenir un résultat final, mais de montrer une méthode claire, reproductible et exploitable.
