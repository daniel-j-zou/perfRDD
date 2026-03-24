# README — Dual-Threshold PLM Mortgage Analysis

## 1. Variable Definitions (Read This First)

This section puts **all main variables and analysis objects at the front** so the project can be understood quickly before reading the workflow.

### 1.1 Core datasets

- **`df`**  
  The cleaned working dataset produced during `data_process.ipynb` after variable conversion, missing-value handling, and basic sample restrictions.

- **`dftest`**  
  The final analysis sample used in the PLM stage. In your workflow, this is the main dataset passed into `plm_analysis_updated.ipynb`. It is a more homogeneous subset of the HMDA data, especially after restricting to single-unit loans.

- **`local`**  
  A temporary local-window sample used in descriptive checks around the thresholds. This is not always a permanently saved dataset; rather, it is a threshold-neighborhood subset used for diagnostics and interpretation.

---

### 1.2 Raw variables used in the final analysis

#### Outcome-related variable

- **`interest_rate`**  
  The stated mortgage interest rate in the HMDA data. This is the base financing-price variable. In the final analysis, it is not used alone as the main outcome; instead, it is adjusted with a PMI proxy to build a broader financing-cost measure.

#### Running variables / threshold variables

- **`combined_loan_to_value_ratio`**  
  The CLTV ratio. This is one of the two running variables in the dual-threshold design. Economically, it captures leverage relative to property value. Around 80, this variable is closely related to PMI-type financing frictions.

- **`loan_amount`**  
  The mortgage loan amount. This is the second running variable. Around the conforming loan limit, this variable is used to study changes in pricing or product structure when loans move into a jumbo-like region.

#### Borrower and contract characteristics

- **`income`**  
  Borrower income. Used as a control variable to adjust for borrower quality and affordability.

- **`debt_to_income_ratio`**  
  Original DTI variable in the raw HMDA data. This variable often appears as strings such as ranges or categories rather than clean numeric values.

- **`dti_num`**  
  A numeric approximation of `debt_to_income_ratio` created in `data_process.ipynb`. This transformed version is what enters the final model.

- **`property_value`**  
  Reported property value. Used as a control because loan pricing depends strongly on collateral value and overall deal structure.

- **`loan_term`**  
  Loan term, typically measured in months. This affects monthly payment structure and overall financing cost, so it is an important loan-contract control.

#### Demographic and product-category controls

- **`derived_ethnicity`**  
  Borrower ethnicity category from HMDA-derived fields.

- **`derived_race`**  
  Borrower race category from HMDA-derived fields.

- **`derived_sex`**  
  Borrower sex category from HMDA-derived fields.

- **`derived_loan_product_type`**  
  Derived loan product category. This helps control for product-design differences that may affect pricing.

- **`occupancy_type`**  
  Indicates whether the property is owner-occupied, second home, investment property, etc.

- **`construction_method`**  
  Indicates whether the dwelling is site-built, manufactured housing, or another type.

- **`loan_type`**  
  Loan category such as conventional, FHA, VA, or other program types.

- **`loan_purpose`**  
  Purpose of the loan, such as home purchase, refinance, or home improvement.

- **`lien_status`**  
  Indicates whether the loan is first lien, subordinate lien, or another lien position.

- **`business_or_commercial_purpose`**  
  Indicator for whether the loan has a business/commercial purpose rather than a standard household mortgage purpose.

- **`open_end_line_of_credit`**  
  Indicator for whether the loan is an open-end line of credit.

- **`hoepa_status`**  
  Indicator related to HOEPA coverage.

- **`reverse_mortgage`**  
  Indicator for reverse-mortgage loans.

#### Geographic / tract-level controls

- **`activity_year`**  
  HMDA activity year.

- **`county_code`**  
  County identifier used as a location control.

- **`tract_population`**  
  Census-tract population.

- **`tract_minority_population_percent`**  
  Share of minority population in the census tract.

- **`ffiec_msa_md_median_family_income`**  
  FFIEC metro-area median family income measure.

- **`tract_to_msa_income_percentage`**  
  Relative tract income compared with the broader MSA/MD area.

- **`tract_owner_occupied_units`**  
  Number of owner-occupied housing units in the tract.

- **`tract_one_to_four_family_homes`**  
  Number of one-to-four family homes in the tract.

- **`tract_median_age_of_housing_units`**  
  Median housing age in the tract.

---

### 1.3 Sample-selection and processing variables

- **`total_units`**  
  Number of units associated with the property. In your analysis, `dftest` is formed by restricting to `total_units == 1`, which keeps the sample focused on single-unit residential loans and reduces heterogeneity from multi-unit properties.

- **Missing-rate threshold (`<= 35%`)**  
  In the processing stage, variables with very high missingness are removed before the final complete-case step. This is part of the data-quality screening logic in `data_process.ipynb`.

- **`dropna()` step**  
  After keeping acceptable variables, the working sample is restricted to complete observations for the retained fields. This creates a cleaner modeling sample at the cost of sample size.

---

### 1.4 Constructed variables used in the analysis notebook

#### Threshold constants and aliases

- **`SCORE1_COL`** = `"combined_loan_to_value_ratio"`  
  The first running variable used in the model.

- **`SCORE2_COL`** = `"loan_amount"`  
  The second running variable used in the model.

- **`C1`** = `80.1`  
  First threshold, corresponding to the CLTV cutoff used in the local dual-threshold analysis.

- **`C2`** = `766550`  
  Second threshold, corresponding to the 2024 conforming loan limit used in the analysis.

#### Outcome construction

- **`PMI_RATE`** = `0.008`  
  Annualized PMI proxy rate, interpreted as 0.8 percentage points when converted into the same units as `interest_rate`.

- **`cltv_gt_80`**  
  Binary indicator defined as:
  `cltv_gt_80 = 1{ combined_loan_to_value_ratio > 80 }`
  This marks observations above the CLTV / PMI-related region.

- **`y`**  
  The final outcome variable used in the PLM analysis:
  `y = interest_rate + 100 * PMI_RATE * cltv_gt_80`
  Since `100 * 0.008 = 0.8`, this means:
  
  - if `CLTV <= 80`, then `y = interest_rate`
  - if `CLTV > 80`, then `y = interest_rate + 0.8`
  
  So `y` should be interpreted as a **broad financing-cost measure**, not just the posted mortgage rate. It combines the observed rate with a PMI-style cost proxy.

- **`Y_COL`** = `"y"`  
  The name of the outcome column passed into the final model.

- **`PLOT_Y_COL`** = `"interest_rate"`  
  The version used in some plots when you want to visualize the raw interest rate rather than the PMI-adjusted outcome.

#### Local-window settings

- **`FIT_WINDOW1`** = `10`  
  Local window around the first threshold. The sample keeps observations satisfying:
  `|combined_loan_to_value_ratio - 80.1| <= 10`

- **`FIT_WINDOW2`** = `100000`  
  Local window around the second threshold. The sample keeps observations satisfying:
  `|loan_amount - 766550| <= 100000`

#### Control-variable container

- **`X_COL`**  
  The list of control variables used to build the model matrix `x_mat`. In your final notebook, this includes borrower characteristics, loan-product variables, and tract-level controls.

- **`x_mat`**  
  The numeric design matrix created from `X_COL` after dummy encoding categorical variables, cleaning invalid values, dropping constant columns, and standardizing the retained columns. This is the linear control block in the partially linear model.

---

### 1.5 Variables created inside the PLM procedure

#### Running-variable recentering

- **`r1`**  
  Recentered first running variable:
  `r1 = score1 - C1`
  In your application:
  `r1 = combined_loan_to_value_ratio - 80.1`
  Positive values mean the observation lies above the CLTV threshold.

- **`r2`**  
  Recentered second running variable:
  `r2 = score2 - C2`
  In your application:
  `r2 = loan_amount - 766550`
  Positive values mean the observation lies above the loan-amount threshold.

#### Treatment definition

- **`treated`**  
  Binary indicator defined using the AND rule:
  `treated = (r1 > 0) & (r2 > 0)`
  Therefore, a loan is treated only if it is simultaneously above both thresholds. Geometrically, this is the upper-right quadrant after centering both running variables at zero.

#### Residualized running variables

- **`etaHat1`**, **`etaHat2`**  
  These are the residualized versions of `r1` and `r2` after projecting them on the control matrix `X`. Economically, they represent the remaining threshold-related variation after adjusting for observed borrower, contract, and location characteristics.
  These are the key nonparametric index variables used by the PLM estimator.

- **`etaTr1`**, **`etaTr2`**  
  Residualized indices for the treated subsample.

- **`etaCon1`**, **`etaCon2`**  
  Residualized indices for the control subsample.



## 2. Project Overview

This project has two stages:

1. **Data processing** in `data_process.ipynb`  
   Clean the HMDA 2024 public LAR data, convert variables, restrict the sample, and build the final analysis dataset `dftest`.

2. **PLM analysis** in `plm_analysis_updated.ipynb` using `plm_dual_threshold.py`  
   Estimate a dual-threshold partially linear model around the CLTV threshold and the conforming-loan-limit threshold.

---

## 3. Data-Processing Summary

The processing notebook mainly does the following:

- load the HMDA raw data;
- remove variables with excessive missingness;
- convert numeric-like strings into usable numeric variables;
- build `dti_num` from `debt_to_income_ratio`;
- convert categorical columns to category type;
- restrict to `combined_loan_to_value_ratio <= 100`;
- restrict to `total_units == 1`;
- create the final analysis sample `dftest`.

---

## 4. Method Summary

The analysis is a **dual-threshold partially linear model (PLM)**.

The two thresholds are:

- **CLTV threshold:** `C1 = 80.1`
- **Loan amount threshold:** `C2 = 766550`

The treatment rule is:

`treated = 1` only when the loan is above **both** thresholds.

The model is estimated only inside a local window around these thresholds, rather than on the full sample.

---

## 5. Local Interpretation

The estimated effect is a **local effect near the two cutoffs**, not a market-wide average effect.

In this README, “local” means two things:

1. the sample is restricted to observations near the thresholds;
2. the smooth component of the PLM is estimated nonparametrically in that local threshold region.

So the final effect should be interpreted as a **near-cutoff financing-cost effect under the dual-threshold design**.

---

## 6. Main Outputs to Report

The most important final outputs are:

- **`ewA_dual`**: estimated local average effect;
- **`cost` / `COST`**: assumed cost benchmark;
- **`net_gain`**: `ewA_dual - cost`;
- **`n_treated`**: number of treated observations;
- **`n_insupp`**: number of treated observations on common support.

These are the quantities that summarize whether the estimated local benefit is economically meaningful under the chosen cost benchmark.

---

## 7. Figures and Diagnostics

The Python file provides several plotting functions, including:

- `plot_results()`
- `plot_component_curves()`
- `plot_alpha_scatter_std()`
- `plot_alpha_heatmap()`
- `plot_fullsample_local()`

These are mainly used to inspect:

- the treated/control geometry near the thresholds;
- the distribution of the estimated local effects;
- how the smooth treated and control components differ;
- whether the effect pattern is stable across the local support.

---

# 
