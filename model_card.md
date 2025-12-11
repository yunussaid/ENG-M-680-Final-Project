
---

# **Model Card — CatBoost Soft Sensor for Silica Prediction**

## **Model Details**

**Model Name:** CatBoost Soft Sensor for % Silica Concentrate
**Model Version:** v1.0
**Model Type:** Regression (tree-based gradient boosting)
**Developers:** Group 2 — ENG M 680
**Release Date:** Fall 2025

---

## **Intended Use**

**Primary Use:**
Provide real-time estimations of **% Silica Concentrate** in a mineral flotation process. The model serves as a *soft sensor*, filling wait time between hourly laboratory measurements.

**Intended Users:**

* Process engineers
* Plant control room operators
* Data scientists building predictive process control systems

**Out-of-Scope Use Cases:**

* Predicting long-term silica grades days or weeks in advance
* Optimization or control algorithms without additional validation
* Use outside iron ore flotation or processes with substantially different sensor layouts

---

## **Model/Data Description**

**Data Used:**
The model is trained on the **Quality Prediction in a Mining Process** dataset (Kaggle).
Key characteristics:

* ~737k samples at **20-second resolution**
* Sensor inputs include pulp flow, pH, density, reagent flows, air flows, and flotation column levels
* Target: **% Silica Concentrate**, measured once per hour
* Preprocessing included datetime parsing, interpolation of missing hour blocks, minute-level resampling, handling rare missing rows, and rolling/lagged feature generation

**Features:**
The final feature set includes:

* Minute-resolution engineered features (lags, rolling statistics, rate-of-change features)
* Physical process signals (flow, pH, reagent dosages, air flows, tank levels)
* Contextual features (hour of day, day of week)

**Model Architecture:**

* **CatBoostRegressor**, depth=6, learning_rate=0.03, n_estimators=300, l2_leaf_reg=3
* Chosen because CatBoost handles non-linear interactions, monotonic drift, and mixed feature scales exceptionally well without explicit normalization
* GPU-accelerated training for efficient tuning

---

## **Training and Evaluation**

**Training Procedure:**

* Chronological **85–15 test/train split** to simulate realistic forward-time generalization
* Hyperparameter tuning via **GridSearchCV** with 3-fold TimeSeriesSplit
* Models trained on fully preprocessed minute-level dataset with engineered features

**Evaluation Metrics:**

* **MAPE:** 0.197
* **MAE:** 0.486
* **R²:** 0.625

**Baseline Comparison:**

* **ElasticNet**: MAPE 0.225, R² 0.647
* **LightGBM**: MAPE 0.240, R² 0.604
  CatBoost was selected because, under a realistic forecasting split, it provided the most stable performance with lower error on minute-level silica estimation.
  Although LightGBM achieved higher performance under a 1–59 per-hour split, that setting evaluates *interpolation,* not generalization. The CatBoost model is therefore more reliable for real-world deployment.

---

## **Ethical Considerations**

**Fairness and Bias:**

* No personally identifiable information is present.
* Bias may arise from uneven distribution of plant operating regimes or sensor faults.
* Model performance may degrade under sensor drift or unobserved maintenance events.

**Privacy:**

* Dataset contains only process measurements; no privacy concerns.

**Security:**

* Model should be deployed only inside secure process control environments.
* Adversarial manipulation of sensor inputs could distort predictions.

---

## **Limitations and Recommendations**

**Known Limitations:**

* Cannot forecast silica beyond short-term trends; relies heavily on engineered lag features.
* Performance may degrade under major changes in ore chemistry or reagent strategy.
* Does not directly model uncertainty; prediction intervals should be added before deployment.

**Recommendations for Use:**

* Use as a decision-support tool, not an automated controller.
* Combine with drift-detection modules to monitor degradation.
* Retrain periodically as new lab data becomes available.

---

## **Additional Information**

**References:**

* “Quality Prediction in a Mining Process” — Kaggle dataset
* Prokhorenkova et al. (2018), CatBoost: unbiased boosting with categorical features
* Google Model Cards (Mitchell et al., 2019)
* References Paper: Machine Learning-based Quality Prediction in the Froth Flotation Process of Mining
* References Paper: Purities prediction in a manufacturing froth flotation plant - the deep learning techniques
* References Paper: Soft Sensor - Traditional Machine Learning or Deep Learning

**License:** Internal academic project; downstream use governed by the dataset’s license.

**Contacts (Group 2 – ENG M 680 Fall 2025):**
* Yunus Said: *ysaid@ualberta.ca*,
* Anu Parbhakar: *aparbhak@ualberta.ca*,
* Anmol Devgun: *devgun@ualberta.ca*,
* Xi Chen: *xc24@ualberta.ca*,
* Youssef Ibrahim: *yibrahim@ualberta.ca*

---
