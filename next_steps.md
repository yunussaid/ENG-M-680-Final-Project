
---

# **Next Steps**

This section outlines opportunities to further improve the soft sensor, strengthen model robustness, and expand the project beyond the scope of the initial milestone.

---

## **1. Explore Sequence Models (TCN / LSTM / GRU)**

Our current approach relies on engineered lag features and tree-based models.
Future work should explore **true sequence architectures**, including:

* **Temporal Convolutional Networks (TCN)**
* **LSTM / GRU recurrent networks**
* **Hybrid CNN–RNN architectures**

These models can capture longer temporal dependencies without manually engineered windows and may generalize better under changing operating conditions.

---

## **2. Expand Evaluation Metrics (MDA & DTW)**

While MAPE and MAE capture magnitude error, they do not measure **directional correctness** or **shape similarity**.

Future evaluation should include:

* **MDA (Mean Directional Accuracy)** — detects whether the model predicts the correct *trend direction*, critical for operators reacting to changes.
* **DTW (Dynamic Time Warping)** — measures the similarity of prediction and actual curves over time, even with temporal shifts.

These additional metrics will provide a more realistic understanding of operational usefulness.

---

## **3. GPU-Accelerated LightGBM Tuning**

LightGBM struggled to run on GPU in the JupyterHub environment, which limited our ability to explore larger hyperparameter grids.

Next steps:

* Install GPU-compatible LightGBM locally or in a controlled cluster environment
* Perform deeper tuning (num_leaves, feature_fraction, min_data_in_leaf, etc.)
* Benchmark GPU performance vs. CPU for large feature sets

This may significantly improve both speed and accuracy.

---

## **4. Incorporate External Datasets**

Integrating contextual variables could boost predictive performance, but sourcing reliable data is challenging because **the plant location is unknown**.

Potential additions include:

* Regional or global weather proxies
* Commodity price or production schedule data
* Ore-type characterization datasets (if source mine is identified)

If plant metadata becomes available, external data alignment could meaningfully improve model generalizability.

---

## **5. Investigate Overly Repeated Sensor Values**

Frequency plots revealed that some sensors produce long stretches of **repeated or constant values**, which may:

* Indicate sensor plateauing or physical limits
* Reflect data logging artifacts
* Reduce model discriminative power

Next step:
Perform experiments that remove or reduce constant-run signals and evaluate how this affects model stability.

---

## **6. Improve Drift Detection & Model Monitoring**

Future iterations should integrate:

* Stronger **concept drift detectors** (ADWIN, Kolmogorov drift tests, PCA reconstruction error)
* Automated retraining triggers
* Per-feature drift diagnostics for proactive maintenance

This is essential for real-world deployment in dynamic plant environments.

---

## **7. Add Prediction Intervals & Uncertainty Estimation**

Operators need not just a prediction but confidence around it.
We should explore:

* Quantile regression (CatBoost supports this natively)
* Conformal prediction
* Bayesian ensembling or MC dropout for neural models

This increases trust and operational safety.

---

## **8. Move to a Fully Modular Python Package**

We have begun organizing reusable components into `src/`, but next steps include:

* Publishing the soft-sensor package locally
* Adding documentation with docstrings and Sphinx
* Building a CLI for training + evaluation
* Adding automated tests for each module

This will improve maintainability for future teams.

---