# Efficacité énergétique des centres de données

Projet de démonstration pour anticiper la charge des serveurs, détecter les risques de points chauds et proposer une consigne de refroidissement plus efficace.

## Objectifs et composants

| Composant | Technique | Fonction |
|---|---|---|
| Prévision de charge | LSTM | Prévoir la charge CPU sur les 30 prochaines minutes |
| Cartes thermiques | CNN | Estimer le risque de point chaud à partir de cartes 2D |
| Pilotage HVAC | Random Forest | Proposer une consigne de soufflage entre 16 et 26 °C |
| Démonstration | Streamlit | Explorer les résultats, les journaux et une décision de contrôle |

Le notebook peut utiliser les jeux Kaggle **Server Workload Traces** et **Data Center Temperature & Humidity Sensors**. Ces données ne sont pas incluses dans le dépôt. Sans fichiers CSV compatibles, le notebook utilise un jeu synthétique reproductible et signale explicitement ce mode.

## Démarrage rapide (Windows PowerShell)

Python 3.12 a été utilisé pour valider l’exécution.

Depuis la racine du dépôt :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-notebook.txt
python -m pip install -r frontend\requirements.txt
```

Ouvrir ensuite `datacenter_cooling_ai.ipynb` dans Jupyter et exécuter les cellules dans l’ordre :

```powershell
python -m notebook datacenter_cooling_ai.ipynb
```

Pour lancer le tableau de bord :

```powershell
Set-Location frontend
python -m streamlit run app.py
```

L’application est disponible à `http://localhost:8501`. Elle recherche les résultats dans `frontend/outputs/`, puis dans `outputs/` à la racine du dépôt. Vous pouvez aussi préciser un autre dossier dans sa barre latérale.

## Utiliser les données Kaggle

1. Télécharger les deux jeux de données Kaggle et placer les CSV dans un dossier local (par exemple `data/`).
2. Dans la cellule `CONFIG` du notebook, renseigner `WORKLOAD_CSV` et `SENSOR_CSV` avec les chemins des fichiers.
3. Exécuter toutes les cellules et vérifier dans la sortie que les fichiers attendus ont été chargés et que le mode synthétique n’a pas été activé.

Le notebook essaie aussi de détecter les fichiers sous `/kaggle/input` lorsqu’il est exécuté sur Kaggle. Si des chemins explicites sont fournis mais que les fichiers sont invalides, l’exécution s’arrête avec l’erreur de lecture au lieu de masquer le problème par un jeu synthétique.

## Fichiers produits

Le notebook crée `outputs/` dans son répertoire de travail (ou `/kaggle/working/` sur Kaggle) :

- `metrics.json` : métriques des modèles et de l’estimation énergétique ;
- `test_control_log.csv` : charge, consigne proposée et puissance de refroidissement sur le test ;
- `lstm_cpu_forecast.keras`, `cnn_hotspot.keras` : modèles de prévision et de classification ;
- `rf_hvac_controller.joblib` : modèle de proposition de consigne.

Le tableau de bord lit ces fichiers. Les artefacts générés ne sont pas nécessaires au code source et sont ignorés par Git.

## Tests

Depuis la racine du dépôt, après installation des dépendances du frontend :

```powershell
python -m unittest discover -s frontend\tests -v
```

## Méthodologie et limites

Les séries sont alignées sur un pas de cinq minutes, puis séparées chronologiquement en ensembles d’entraînement, validation et test. Les jeux Kaggle peuvent provenir de salles et de périodes différentes ; leur alignement ne constitue pas une mesure de causalité physique.

Les sources publiques ne fournissent pas toujours les coordonnées physiques des capteurs ni les actions HVAC réelles. Le Random Forest apprend donc à imiter une politique oracle basée sur un modèle thermique simplifié. L’énergie, le PUE, les coûts et les émissions sont des estimations paramétrables, pas des mesures d’un site.

Les prédictions LSTM et CNN utilisées comme variables d’entraînement du Random Forest sont produites par des modèles ajustés sur la période d’entraînement. Elles sont donc en partie « in-sample » ; une validation chronologique walk-forward est nécessaire avant d’interpréter les résultats comme une performance de généralisation.

**Cette démonstration ne commande pas d’équipement.** Toute utilisation réelle nécessiterait une validation avec des mesures HVAC, des limites de sécurité validées par des spécialistes et un mécanisme de retour à une consigne sûre.

## Livrables

- [Notebook de modélisation](./datacenter_cooling_ai.ipynb)
- [Rapport technique](./Rapport%20technique%20%E2%80%94%20Efficacit%C3%A9%20%C3%A9nerg%C3%A9tique%20des%20centres%20de%20donn%C3%A9es.docx)
- [Interface Streamlit](./frontend/app.py)
- [Instructions du frontend](./frontend/README.md)
