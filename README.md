# Silica Concentrate Soft Sensor Project

## Overview
This project implements a *Machine Learning Soft Sensor* to predict *% Silica Concentrate* in a mining flotation process. By analyzing real-time process variables (air flow, level, pulp density, etc.), the system predicts silica levels every minute, allowing operators to react to process changes immediately rather than waiting 60 minutes for physical lab results.

The final production model is built using *CatBoost*, selected for its superior generalization capability on future data.

## Project Structure
- soft_sensor.ipynb: Contains the production training pipeline.
- research/: Archive of EDA of mining and other dataset.
- Aarchitecture_diagrams.png: visuals on  how the system works (where it pulls data from, etc.)

## Usage
1. Install dependencies: pip install -r requirements.txt
2. Run training: jupyter notebook notebooks/soft_sensor.ipynb

## Key Performance (Forward-in-Time Evaluation)
We prioritized the *85–15 Chronological Split* for model selection to simulate true production conditions (predicting real-time % Silica Concentrate in future use).
- *Model:* CatBoost Regressor
- *MAPE:* 19.7%
- *R²:* 0.62
- *MAE:* 0.48

> *Note on Evaluation Strategy:*
> The 85–15 split represents true forward-in-time generalization. While other models (like LightGBM) showed inflated scores (R² ~0.91) under a 1–59 split, those metrics rely on interpolating within the same hour where the lab target is constant. 
>
> We chose the 85–15 split results because they reflect the actual difficulty of predicting silica across different unseen months (not in trainig data) and operating regimes, providing a realistic expectation of performance in a live plant.



