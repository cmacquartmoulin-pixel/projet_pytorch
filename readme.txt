Ordre d'exécution :

1) src/train.py: entraîne le modèle (charge le dataset automatiquement)
2) src/test.py: teste le modèle avec les poids issus de l'entraînement
3) src/optuna_search.py: optimisation automatique des hyperparamètres

Pour visualiser les résultats, ouvrir un terminal dans le dossier racine du projet et exécuter:
   optuna-dashboard sqlite:///optuna_cifar10.db

L'architecture du réseau est configurable via conf/config.yaml