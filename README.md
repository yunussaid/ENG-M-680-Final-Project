# Silica Concentrate Soft Sensor Project

## Overview
This project implements a *Machine Learning Soft Sensor* to predict *% Silica Concentrate* in a mining flotation process. By analyzing real-time process variables (air flow, level, pulp density, etc.), the system predicts silica levels every minute, allowing operators to react to process changes immediately rather than waiting 60 minutes for physical lab results.

The final production model is built using *CatBoost*, selected for its superior generalization capability on future data.

> [*Note: Refer to our other repository to view our implemented dashboard*](https://github.com/yunussaid/ENG-M-680-Project)

## Project Structure
| File / Directory | Description |
|------------------|-------------|
| **`soft_sensor.ipynb`** | **Final Training Notebook:** Clean, modular notebook containing the full pipeline for feature engineering, model training, hyperparameter tuning, and evaluation of the soft sensor. |
| **`model_card.md`** | **Model Documentation:** A comprehensive model card detailing the final chosen CatBoost model's performance metrics, intended use, limitations, ethical considerations, and training parameters. |
| **`next_steps.md`** | **Future Work:** Document outlining potential next steps, including exploring sequence models, advanced metrics (MDA/DTW), GPU tuning, external data integration, and drift analysis. |
| **`README.md`** | **Project Overview:** Main documentation describing the project goal, setup instructions, methodology, and repository structure. |
| **`architecture_diagrams.png`** | **Visuals:** High-level system and deployment diagrams illustrating how the soft sensor integrates into the mining data pipeline. |
| **`library/`** | **Python Library:** Python library version `soft_sensor_lib.py` of the pipeline for feature engineering, model training, hyperparameter tuning, and evaluation of the soft sensor (+ function calls from `soft_sensor_lib.ipynb` notebook). |
| **`requirements.txt`** | **Dependencies:** List of Python packages required to run the project, ensuring reproducibility and environment consistency. |
| **`datasets/`** | **Raw Data Storage:** Folder containing the unmodified dataset sources used for analysis and model development (the Kaggle flotation dataset ZIP). |
| **`datasets/quality_prediction_in_a_mining_process.zip`** | **Original Dataset Archive:** The raw ZIP file downloaded from Kaggle; extracted and processed by the notebook. |
| **`research/`** | **Exploratory Work:** Contains development and analysis notebooks not intended for final evaluation, documenting EDA and experimentation. |
| **`research/initial_eda.ipynb`** | **Early Exploration:** Notebook performing initial exploratory data analysis, distribution checks, and dataset validation. |
| **`research/inspect_all_datasets.ipynb`** | **Supplementary Analysis:** Notebook used to explore other datasets the team was considering doing a project on. |

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `jupyter notebook soft_sensor.ipynb`

## Key Performance (Forward-in-Time Evaluation)
We prioritized the *85–15 Chronological Split* for model selection to simulate true production conditions (predicting real-time % Silica Concentrate in future use).
- *Model:* CatBoost Regressor
- *MAPE:* 19.7%
- *R²:* 0.62
- *MAE:* 0.48

> *Note on Evaluation Strategy:*
> The 85–15 split represents true forward-in-time generalization. While other models (like LightGBM) showed inflated scores (R² ~0.91) under a 1–59 split, those metrics rely on more so interpolating within the same hour where the lab target is constant and seen during training. 
>
> We chose the 85–15 split results for picking our final model because they reflect the actual difficulty of predicting silica across different months and operating regimes (that are not seen during trainig), providing a realistic expectation of real-time performance in a live plant.



