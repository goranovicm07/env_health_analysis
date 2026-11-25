# Environmental Contaminants, Life Expectancy, and Cancer Incidence Rates

## Executive Summary
This project investigates how environmental contaminants (along with agricultural land use, and socioeconomic / demographic factors) relate to life expectancy (`Total`) and unadjusted cancer rates (`Crude Rate`) across all 72 counties in Wisconsin. Using dimensionality reduction (`|r| >= 0.999` cluster deduplication from 218 to 196 features), Bonferroni-adjusted bivariate screening, and 5-fold cross-validated regularized regression (`RidgeCV`, `LassoCV`, `ElasticNetCV`) benchmarked against unregularized OLS and a regularized `RandomForestRegressor`, this project separates environmental features from demographic confounding.

**Core Findings:**
- **Life Expectancy (`Total`):** Adding environmental and water-quality predictors to socioeconomic controls substantially improves out-of-sample generalization (`ElasticNetCV` 5-fold CV $R^2$ rises from **0.4158** to **0.6473**, reducing RMSE from **1.10** to **0.86 years**).
- **Cancer (`Crude Rate`):** Demographic and housing structure alone explains most predictable variance (`RidgeCV` 5-fold CV $R^2 = \mathbf{0.6938}$), while adding 184 environmental predictors slightly degrades generalization due to noise and age-structure confounding.
- **Cross-Outcome Relationship:** Life expectancy and crude cancer rates show a moderate negative bivariate correlation (Pearson $r = -0.3148$, Spearman $\rho = -0.4406$) that vanishes once both confounders and environmental features are jointly controlled ($r = -0.0703, p = 0.557$), demonstrating that the two outcomes respond to distinct demographic and spatial regimes rather than a single shared environmental driver.

---

## Problem Statement
Public health and environmental datasets frequently exhibit severe high-dimensional multicollinearity ($p > n$) alongside strong spatial and socioeconomic confounding. Evaluating raw correlations between environmental pollutants and county health metrics can produce misleading conclusions if demographic structure (such as unemployment or poverty) is ignored.

This study addresses three research questions across Wisconsin's 72 counties:
1. **RQ1**: Is there a significant relationship between environmental contaminants and life expectancy in Wisconsin counties?

2. **RQ2**: Is there a significant relationship between environmental contaminants and cancer rates in Wisconsin counties?

3. **RQ3**: What is the relationship, if any, between life expectancy and cancer rates based on environmental factors?

---

## Methodology
1. **Data Cleaning & Dimensionality Reduction ($p > n$ Mitigation):**
   - Evaluated 72 Wisconsin counties across 218 active numeric predictors spanning air pollutants (`a_*`), water contaminants (`w_*`), agricultural / pesticide metrics, built-environment rates, and socioeconomic / housing controls.
   - Identified near-identical feature clusters ($|r| \ge 0.999$) and retained 1 representative feature per cluster, reducing the feature space to 196 predictors (12 socioeconomic / housing confounders and 184 environmental representatives).
   - *Note*: Strong multicollinearity  still remained.
2. **Exploratory Data Analysis & Multiple-Testing Control:**
   - Conducted bivariate Pearson / Spearman screening with Bonferroni correction ($\alpha_{\text{adj}} \approx 2.29 \times 10^{-4}$) to control false discoveries.
   - Identified high-leverage bivariate outliers, including Menominee County (lowest life expectancy at $\approx 72.5$ years and $\ge 16\%$ unemployment, yet a moderate crude cancer rate of $\approx 481$) and northern recreational / retirement counties (Vilas, Iron, Adams, Oneida) with elevated crude cancer rates ($>800$ to $>1,000$).
3. **Controlled & Domain-Grouped Modeling Pipeline:**
   - Built leak-free `scikit-learn` pipelines (`StandardScaler` + estimator) evaluated via 5-fold cross-validation (`KFold`, `shuffle=True`, `random_state=24`).
   - Benchmarked unregularized OLS (demonstrating severe $p > n$ overfitting) against RidgeCV, LassoCV, ElasticNetCV, and a regularized `RandomForestRegressor` across five controlled feature sets:
     1. `1_Confounders_Only` (12 features)
     2. `2_Confounders_+_All_Env` (196 features)
     3. `3_Confounders_+_Air` (97 features)
     4. `4_Confounders_+_Water` (51 features)
     5. `5_Confounders_+_Ag_Land` (21 features)
4. **Cross-Target Residual & Coefficient Analysis (RQ3):**
   - Computed out-of-fold residual correlations between `Total` and `Crude Rate` across feature sets and compared standardized non-zero `ElasticNetCV` coefficients across both targets.

---

## Skills & Tools
- **Languages & Core Libraries:** Python, `pandas`, `numpy`, `scipy` (`stats`)
- **Machine Learning & Statistical Modeling:** `scikit-learn` (`Pipeline`, `StandardScaler`, `KFold`, `cross_val_predict`, `LinearRegression`, `RidgeCV`, `LassoCV`, `ElasticNetCV`, `RandomForestRegressor`)
- **Statistical Techniques:** High-dimensional feature deduplication ($|r| \ge 0.999$), Bonferroni multiple-hypothesis testing, L1 / L2 regularization, out-of-fold residual analysis, leverage & outlier diagnostics
- **Data Visualization:** `matplotlib`, `seaborn` (bivariate trend scatterplots, county ranking bar charts, CV $R^2$ comparisons, out-of-fold predicted vs. actual diagnostics, regularized coefficient plots)

---

## Results & Recommendations

### Model Performance Summary (5-Fold Cross-Validated $R^2$)
| Target | Best Feature Set | Best Model | 5-Fold CV $R^2$ | CV RMSE | Key Insight |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Life Expectancy (`Total`)** | `2_Confounders_+_All_Env` (196) | `ElasticNetCV` | **0.6473** | **0.8577** | Environmental & water predictors (`0.6386`) add substantial lift over confounders alone (`0.4158`). |
| **Cancer (`Crude Rate`)** | `1_Confounders_Only` (12) | `RidgeCV` | **0.6938** | **65.5074** | Demographic & housing controls explain the majority of predictable variance; extra env noise lowers CV $R^2$. |
| **Unregularized Baseline** | `rep_features` (196) | `LinearRegression` | 0.2781 / -0.3965 | 1.23 / 139.89 | Memorizes training data (in-sample $R^2 = 1.00$) and collapses out of sample. |

### Recommendations
- **Target Socioeconomic & Water-Quality Interventions Jointly for Longevity:** Because socioeconomic distress (`pct_unemp`, `pct_hs_more`, `med_rooms`) and water / chemical indicators jointly improve life expectancy prediction, county public health initiatives should pair economic support with targeted water-quality monitoring.
- **Avoid Interpreting Unadjusted Cancer Rates Without Demographic Controls:** Public health reporting and resource allocation should rely on age-adjusted rather than crude cancer rates to avoid conflating retirement migration (`pct_vac_units`) with environmental hazard.
- **Prefer Penalized Linear Models Over Tree Ensembles on Small-$N$ Tabular Health Data:** Across both outcomes, `ElasticNetCV` and `RidgeCV` substantially outperformed `RandomForestRegressor` (max CV $R^2 = 0.3630$ for `Total` and $0.5057$ for `Crude Rate`), making regularized linear models the recommended baseline for county-level ecological modeling.

---

## Limitations & Next Steps
- **Unadjusted Cancer Rates:** `Crude Rate` is not age-adjusted; future iterations should incorporate age-adjusted cancer incidence / mortality rates and explicit median-age covariates.
- **Ecological Aggregation ($N = 72$):** County-level averages cannot capture individual exposure doses, occupational risks, or sub-county disparities (e.g., census-tract or private-well variation).
- **Cross-Sectional & Proxy Constraints:** Multi-year cancer latency requires longitudinal exposure lags, and remaining collinearity among chemical / agricultural indicators means selected Lasso / ElasticNet features act as proxies for broader regional exposure clusters.
- **Next Steps:** Incorporate multi-year lagged EPA / CDC environmental panels, evaluate spatial autocorrelation models (Spatial Lag / Spatial Error), and test age-adjusted mortality outcomes.