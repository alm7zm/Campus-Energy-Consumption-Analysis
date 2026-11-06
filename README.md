# Campus Energy Consumption Analysis

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Model](https://img.shields.io/badge/Model-XGBoost-189AB4)
![Data](https://img.shields.io/badge/Data-UNICON-2E8B57)

Statistical analysis and machine learning on **1.25 million energy meter readings** from three university campuses. The project compares the campuses, tests what drives energy use (weather, weekdays, business hours, COVID-19), and builds campus-specific XGBoost models to predict consumption.

This repository contains **Phase 3** of the project: hypothesis testing, campus comparison and predictive modelling, run on the cleaned data prepared in earlier phases.

---

## Key Findings

- **Accurate campus-specific models:** XGBoost reaches **R² = 0.81–0.90** (average 0.86) on the test set, above the 0.765 reference R² of a combined all-campus model on Kaggle.
- **Building identity matters most:** `meter_id` accounts for **52%** of feature importance (Campus 1 model). Adding it along with engineered features lifts R² from 0.03–0.19 to 0.81–0.90.
- **COVID-19 cut consumption:** average consumption fell **31.6%** after March 2020, from −22% on Campus 1 to −69% on Campus 2.
- **High consumption is mostly off-hours:** **58.5%** of high-consumption readings happen outside business hours (9 AM–5 PM).
- **Weather and weekdays have small effects:** weekend consumption is 6.9% lower than on weekdays, and temperature on its own explains only **0.12%** of the variation.

---

## Dataset

The data comes from **UNICON**, an open dataset of electricity, gas and water consumption across La Trobe University's campuses in Australia, published by the university's Centre for Data Analytics and Cognition (CDAC).

- Kaggle: https://www.kaggle.com/datasets/cdaclab/unicon
- GitHub: https://github.com/CDAC-lab/UNICON

| Campus | Readings | Buildings (meters) | Avg consumption | Avg per building |
|---|---:|---:|---:|---:|
| Campus 1 | 963,706 | 60 | 34.83 kWh | 45.16 kWh |
| Campus 2 | 188,407 | 11 | 7.19 kWh | 5.93 kWh |
| Campus 3 | 102,521 | 7 | 23.54 kWh | 16.84 kWh |
| **Total** | **1,254,634** | **78** | | |

The cleaned readings span **January 2015 to December 2022**.

The notebook expects three cleaned CSV files (`campus1.csv`, `campus2.csv`, `campus3.csv`) with these columns:

| Column | Description |
|---|---|
| `timestamp` | Date and time of the reading |
| `meter_id` | Building / meter identifier |
| `consumption` | Energy consumption (kWh) |
| `air_temperature` | Air temperature (°C) |
| `relative_humidity` | Relative humidity |
| `wind_speed` | Wind speed |
| `hour`, `dayofweek`, `month` | Time features |
| `is_weekend` | 1 = weekend, 0 = weekday |

> **Note:** The CSV files are not included in this repository.

---

## Analysis Overview

The notebook [`PHASE3_UNICON_FINALSTEPS (1).ipynb`](PHASE3_UNICON_FINALSTEPS%20%281%29.ipynb) is organized into five parts:

1. **Campus comparison:** raw vs. per-building average consumption
2. **Hypothesis testing:** ANOVA, Pearson correlation and t-tests
3. **Bayesian probability:** when does high consumption happen?
4. **Machine learning:** an XGBoost model for each campus, with feature importance
5. **COVID-19 impact:** before vs. during the pandemic, overall and by campus

### 1. Campus Comparison

Campus 1 has the highest consumption and Campus 2 the lowest, both overall and per building. Campus 2's lower usage likely reflects smaller buildings rather than better efficiency.

### 2. Hypothesis Testing (α = 0.05)

| Research question | Test | Result | Decision |
|---|---|---|---|
| Do the three campuses differ in consumption? | One-way ANOVA | F = 6,241.22, p < 0.0001 | Reject H₀ |
| Does air temperature affect consumption? | Pearson correlation | r = 0.034, p < 0.0001 (R² = 0.12%) | Reject H₀, but the effect is very weak |
| Do weekdays and weekends differ? | Independent t-test | 30.35 vs. 28.26 kWh (weekends 6.9% lower), t = 10.49, p < 0.0001, Cohen's d = 0.021 | Reject H₀, but the effect is very small |
| Did consumption change during COVID-19? | Independent t-test | 35.65 → 24.38 kWh (−31.6%), t = 17.69, p < 0.0001 | Reject H₀ |

With 1.25 million readings, even tiny differences become statistically significant, so effect sizes (r, R², Cohen's d) are reported alongside p-values.

### 3. Bayes' Theorem: When Does High Consumption Occur?

*If a reading is high, how likely is it to fall within business hours?*

- **High consumption:** top 25% of readings (above 19.38 kWh)
- **Business hours:** 9 AM to 5 PM

| Probability | Value |
|---|---:|
| P(Business hours) | 0.367 |
| P(High consumption) | 0.250 |
| P(High consumption \| Business hours) | 0.282 |
| **P(Business hours \| High consumption)** | **0.415** |

Business hours make up 36.7% of all readings but 41.5% of high-consumption readings. That leaves **58.5% of high consumption outside business hours**, a clear target for scheduling and automated controls.

### 4. Machine Learning: XGBoost

A separate XGBoost regressor is trained for each campus and tested with two feature sets:

| Feature set | Features |
|---|---|
| **Basic** (7) | `air_temperature`, `relative_humidity`, `wind_speed`, `hour`, `dayofweek`, `is_weekend`, `month` |
| **Enhanced** (15) | Basic + cyclical time encodings (`hour_sin`, `hour_cos`, `month_sin`, `month_cos`), `temp_humidity` (temperature × humidity), `feels_like` (temperature − 0.7 × wind speed), `day`, `meter_id` |

**Setup:** 80/20 train/test split (`random_state=42`), standardized features, `n_estimators=300`, `max_depth=10`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`, `tree_method='hist'`.

**Results on the test set:**

| Campus | Buildings | R² (Basic) | R² (Enhanced) | RMSE (Enhanced) | MAE (Enhanced) |
|---|---:|---:|---:|---:|---:|
| Campus 1 | 60 | 0.029 | **0.874** | 40.47 kWh | 11.15 kWh |
| Campus 2 | 11 | 0.191 | **0.811** | 5.43 kWh | 2.47 kWh |
| Campus 3 | 7 | 0.147 | **0.902** | 8.87 kWh | 4.64 kWh |
| **Average** | | 0.122 | **0.862** | | |

All three models score above the **R² = 0.765** reference from a combined all-campus model on Kaggle. That model used a different setup (one model for all campuses), so treat it as a reference point rather than a head-to-head comparison.

**Top features (Campus 1, Enhanced model):**

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `meter_id` | 52.1% |
| 2 | `wind_speed` | 16.5% |
| 3 | `is_weekend` | 4.4% |
| 4 | `month` | 3.7% |
| 5 | `month_cos` | 3.2% |

Building identity dominates: much of the variation in consumption comes down to *which* building is being measured. Air temperature ranks low (1.6%), matching the weak correlation found in the hypothesis tests.

### 5. COVID-19 Impact

Readings are split at **1 March 2020** into pre-COVID (Jan 2015 – Feb 2020, 598,212 readings) and the COVID period (Mar 2020 – Dec 2022, 656,422 readings).

| Campus | Pre-COVID avg | COVID-period avg | Change |
|---|---:|---:|---:|
| Campus 1 | 39.06 kWh | 30.63 kWh | −21.6% |
| Campus 2 | 13.69 kWh | 4.25 kWh | −69.0% |
| Campus 3 | 29.94 kWh | 14.48 kWh | −51.6% |
| **All campuses** | **35.65 kWh** | **24.38 kWh** | **−31.6%** |

The drop is statistically significant (t = 17.69, p < 0.0001). Campus 2 saw the largest reduction, likely reflecting extensive closures or a shift to remote operations, while Campus 1 was least affected.

---

## Recommendations

- **Target off-hours consumption:** 58.5% of high readings occur outside 9 AM–5 PM, so scheduling and automated controls could cut waste.
- **Use occupancy-based controls:** the COVID-period drop shows buildings can run at much lower loads when occupancy falls.
- **Monitor building by building:** `meter_id` is the strongest predictor, so per-building baselines and anomaly detection are more useful than campus-wide averages.
- **Forecast per campus:** campus-specific models capture local patterns and can support demand planning.

---

## Limitations & Future Work

- The train/test split is random rather than chronological, so test readings sit between training readings from the same buildings. A time-based split (e.g., train on 2015–2021, test on 2022) would better measure true forecasting accuracy.
- MAPE is computed in the notebook but left out here, because near-zero readings inflate it to meaningless values.
- The COVID-19 comparison is a simple before/after split and doesn't adjust for other changes over time, such as meters being added or removed.
- Adding building metadata (floor area, building type) could explain *why* buildings differ, not just *that* they do.

---

## Repository Structure

```
Campus-Energy-Consumption-Analysis/
├── PHASE3_UNICON_FINALSTEPS (1).ipynb   # Full analysis with outputs and charts
└── README.md
```

## Getting Started

**Requirements:** Python 3 (built with Python 3.13) and Jupyter.

1. **Clone the repository**
   ```bash
   git clone https://github.com/alm7zm/Campus-Energy-Consumption-Analysis.git
   cd Campus-Energy-Consumption-Analysis
   ```
2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost jupyter
   ```
3. **Add the data:** place `campus1.csv`, `campus2.csv` and `campus3.csv` (columns listed under [Dataset](#dataset)) in the same folder as the notebook.
4. **Run the notebook**
   ```bash
   jupyter notebook "PHASE3_UNICON_FINALSTEPS (1).ipynb"
   ```
   Then run all cells from top to bottom.

## Tech Stack

Python · pandas · NumPy · Matplotlib · seaborn · SciPy · scikit-learn · XGBoost · Jupyter

## Acknowledgements

Data: the **UNICON** dataset from La Trobe University's Centre for Data Analytics and Cognition (CDAC).

> H. Moraliyage, N. Mills, P. Rathnayake, D. De Silva, and A. Jennings, "UNICON: An Open Dataset of Electricity, Gas and Water Consumption in a Large Multi-Campus University Setting," *2022 15th International Conference on Human System Interaction (HSI)*, 2022. https://doi.org/10.1109/HSI55341.2022.9869498

## Author

**alm7zm** · [GitHub](https://github.com/alm7zm)
