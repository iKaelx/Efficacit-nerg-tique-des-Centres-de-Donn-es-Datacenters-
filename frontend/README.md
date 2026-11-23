# Efficacité énergétique des centres de données

Projet d'intelligence artificielle pour optimiser le contrôle du refroidissement d'un centre de données. Le notebook combine la prévision de charge serveur, la détection thermique et un contrôleur HVAC afin d'estimer les économies d'énergie, de coût et d'émissions de CO2 par rapport à une consigne fixe.

**Notebook principal :** [datacenter_cooling_ai.ipynb](./datacenter_cooling_ai.ipynb)  
**Rapport technique :** [Rapport technique - Efficacité énergétique des centres de données.docx](./Rapport%20technique%20%E2%80%94%20Efficacit%C3%A9%20%C3%A9nerg%C3%A9tique%20des%20centres%20de%20donn%C3%A9es.docx)

## Fonctionnement

Le projet comprend trois modèles et une évaluation énergétique :

| Composant | Technique | Objectif |
|---|---|---|
| Prévision | LSTM | Prédire la charge CPU des serveurs sur les 30 prochaines minutes |
| Analyse thermique | CNN | Lire une carte thermique 2D des racks et estimer le risque de point chaud |
| Contrôle HVAC | Random Forest | Choisir dynamiquement la température de soufflage |
| Évaluation | Modèle énergétique simplifié | Estimer les kWh, coûts et émissions économisés |

Les données sont alignées à un pas de temps de 5 minutes. La série est divisée chronologiquement en ensembles d'entraînement, de validation et de test (70 % / 15 % / 15 %).

## Données

Le notebook peut utiliser deux jeux de données Kaggle :

- des traces de charge ou d'utilisation des serveurs ;
- des mesures de température et d'humidité du centre de données.

Les chemins peuvent être définis avec `WORKLOAD_CSV` et `SENSOR_CSV`. Sinon, le notebook tente de détecter automatiquement les fichiers CSV disponibles dans `/kaggle/input`.

Si aucun jeu de données réel n'est trouvé ou exploitable, le notebook bascule automatiquement sur un jeu synthétique reproductible. Il produit ainsi une démonstration complète avec une charge quotidienne, une zone chaude fixe dans une salle de racks et des capteurs virtuels.

## Exécution sur Kaggle

1. Importer le notebook dans Kaggle.
2. Ajouter les deux jeux de données CSV.
3. Activer éventuellement un accélérateur GPU T4.
4. Exécuter toutes les cellules.

La sortie locale ou Kaggle est placée dans :

- `/kaggle/working` sur Kaggle ;
- `./outputs` lors d'une exécution locale.

## Exécution locale

Créer un environnement Python puis installer les bibliothèques nécessaires :

```bash
python -m venv .venv
source .venv/bin/activate
pip install notebook numpy pandas matplotlib tensorflow scikit-learn
jupyter notebook datacenter_cooling_ai.ipynb
```

Sous Windows PowerShell, activer l'environnement avec :

```powershell
.venv\Scripts\Activate.ps1
```

Pour utiliser vos propres données, affectez les chemins CSV dans la section de configuration du notebook :

```python
WORKLOAD_CSV = "./data/workload.csv"
SENSOR_CSV = "./data/sensors.csv"
```

## Résultats produits

Le notebook affiche notamment :

- la charge CPU, la température des racks et l'humidité ;
- les performances du LSTM ;
- les performances du CNN sur les points chauds ;
- l'importance des variables du Random Forest ;
- la comparaison entre le contrôle dynamique et une consigne fixe ;
- une estimation des économies d'énergie, de coût et de CO2.

Il génère également :

- `outputs/metrics.json` : indicateurs d'évaluation ;
- `outputs/test_control_log.csv` : journaux de consigne, refroidissement et consommation sur l'ensemble de test.

## Hypothèses et limites

- Les données publiques ne contiennent généralement pas d'historique réel des actions HVAC.
- Le Random Forest apprend donc à reproduire une politique d'oracle calculée avec les températures futures.
- L'énergie, le COP du système de refroidissement et les émissions sont estimés à partir de paramètres simplifiés configurables dans le notebook.
- Les températures, capteurs et éventuels points chauds peuvent être simulés ou reconstruits à partir des colonnes disponibles.
- Les résultats constituent une démonstration technique, pas une validation physique d'un centre de données réel.
- Pour une mise en production, les modèles doivent être validés sur une période indépendante avec des mesures HVAC réelles.
