#!/bin/bash

# Démarrer Uvicorn en arrière-plan
uvicorn api:app --host 0.0.0.0 --port 8000 --reload &

# Démarrer l'interface Python
python interface.py