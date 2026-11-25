# Démonstration Streamlit

## 1. Générer les résultats

Depuis la racine du dépôt, exécuter `datacenter_cooling_ai.ipynb` dans l’ordre. Le notebook crée les fichiers dans `outputs/` à la racine de son répertoire de travail.

## 2. Installer et lancer l’interface

Depuis ce dossier :

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Ouvrir `http://localhost:8501`. L’interface détecte `frontend/outputs/` ou `../outputs/` ; un autre chemin peut être indiqué dans la barre latérale. Elle requiert `metrics.json` et `test_control_log.csv` ; le simulateur de décision en direct utilise aussi `rf_hvac_controller.joblib`.

Si les résultats sont synthétiques, un avertissement est affiché dans l’application. Les sorties sont des estimations de démonstration, pas des consignes à appliquer directement à une installation réelle.
