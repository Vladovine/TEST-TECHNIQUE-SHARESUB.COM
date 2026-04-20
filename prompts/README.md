# Prompts IA — Risk Monitor

Ce dossier contient les prompts versionnés utilisés par l’agent IA de l’application.

L’objectif est de garder une trace claire des itérations du comportement de l’agent, de ses rôles et de ses contraintes de sortie.

## Fichiers

### `agent_analyst_v1.txt`
Prompt du rôle **analyste**.

Ce prompt demande à l’agent de :
- produire un résumé structuré du subscriber ;
- relever les signaux utiles ;
- rester concret et court ;
- répondre uniquement en JSON valide.

### `agent_decider_v1.txt`
Prompt du rôle **décideur**.

Ce prompt demande à l’agent de :
- proposer une recommandation d’action ;
- choisir entre `watch`, `block` ou `ignore` ;
- fournir une justification courte ;
- répondre uniquement en JSON valide.

## Logique de versionnement

Les prompts sont versionnés pour :
- suivre les améliorations ;
- comparer les comportements ;
- documenter les choix de formulation ;
- garder un historique lisible dans le dépôt.

La convention actuelle est :
- `v1` : première version de travail ;
- une nouvelle version est créée si la formulation ou la structure de sortie change de façon significative.

## Utilisation

Ces prompts sont lus par `src/agent.py` et utilisés dans l’interface Streamlit lorsqu’un opérateur ouvre la fiche détaillée d’un subscriber.

## Règle générale

Les prompts doivent rester :
- courts ;
- explicites ;
- orientés sortie structurée ;
- compatibles avec un usage opérationnel.
