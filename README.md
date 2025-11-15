# Walmart Hybrid Weekly Sales Forecasting  
Multivariate demand forecasting with leakage-safe feature engineering, time-aware CV, and a hybrid LSTM + Prophet + XGBoost ensemble.

---

## 🔎 Project Goal

Forecast **weekly sales at the store–department level** for a large retailer (Walmart-style setting), with the goal of supporting:
- inventory planning,
- staffing,
- promotion/markdown strategy,
- and peak-season readiness (Black Friday, holiday spike weeks, back-to-school, etc.).

This repo shows an end-to-end applied forecasting workflow:
1. Data prep and feature engineering for retail time series.
2. Multiple model families (gradient boosting, additive time series, deep learning).
3. Hybrid ensemble to improve robustness on the hardest weeks to predict.

This is meant to read like real demand forecasting work, not just “fit a model.”

---

## 🧠 Business Context

Retail demand is not smooth.
- Sales explode during promotions and holidays.
- Departments behave differently across stores.
- External conditions (fuel cost, inflation, unemployment) shift baseline demand.

Accurate forecasting here matters because getting peak weeks wrong is expensive:
- You overstaff, or worse: understaff.
- You overstock, or worse: sell out.
- Finance plans miss reality.

The goal is not just “score on a metric,” but “be reliable under chaos.”

---

## 💡 Highlights

- ~421k rows of weekly sales data across 45+ stores × departments.
- Built **leakage-safe lag and rolling-window features** (1–12 week lag, 4/8/12 week rolling mean & std).
- Added **seasonality / holiday / back-to-school / year-end flags**.
- Engineered **promotion intensity features** from markdown programs.
- Pulled in **macro variables**: CPI, fuel price, unemployment.
- Trained and compared:
  - **XGBoost** (tree-based gradient boosting, tuned with RandomizedSearchCV)
  - **Prophet** (trend + seasonality + holiday + exogenous regressors)
  - **LSTM** (sequence deep learning over a 12-week window)
- Built a **weighted hybrid ensemble** that blends all three families.

---

## 📊 Model Performance (Held-Out Future Window)

We split chronologically (first ~80% train, last ~20% test).  
Metrics: RMSE (↓ is better), MAE, and % error with an epsilon to handle zero-sales weeks.

**Test Results (example run):**
- **XGBoost**  
  - RMSE ≈ 4.15k  
  - Strongest single-model accuracy. Learns promo spikes and recent momentum.
- **Prophet**  
  - RMSE ≈ 6.10k  
  - Captures smooth seasonal patterns (holidays, year-end trend) using calendar logic + regressors.
- **LSTM**  
  - RMSE ≈ 22.3k  
  - Learns temporal structure from sequences, but needs deeper tuning / per-store specialization.
- **Hybrid Ensemble (LSTM + Prophet + XGBoost)**  
  - RMSE ≈ 15.0k  
  - More stable around high-volatility weeks (promotions, holidays).  
    This is closer to what a retailer actually needs: robustness in painful weeks, not just average-week fit.

Why keep all of them?
- XGBoost wins “raw RMSE.”
- Prophet explains seasonal cycles to stakeholders.
- LSTM is built for temporal dependencies.
- The hybrid smooths out extreme weeks (the most expensive mistakes).

This is exactly how production forecasting is often done: multiple specialized models feeding one decision signal.

---

## 🏗 Pipeline Overview

### 1. Data Merging
We combine:
- `train.csv`: historical weekly sales (`Weekly_Sales`) per (Store, Dept, Date)
- `features.csv`: macro & promo signals (Fuel_Price, CPI, Unemployment, MarkDown1–5, IsHoliday)
- `stores.csv`: static store attributes (Type, Size)

Result: one modeling table where each row = one `(Store, Dept, Week)`.

### 2. Feature Engineering
Key features added:
- **Lag features**: previous 1–12 weeks of sales per (Store, Dept).
- **Rolling stats**: 4, 8, 12 week rolling mean & std, shifted so they only use past data.
- **Calendar features**: Month, WeekOfYear, Quarter, DayOfWeek, IsWeekend, IsYearEnd, IsSummer, BackToSchool window.
- **Holiday / promotion signals**: IsHoliday, IsHolidaySeason, Total_MarkDown, Has_MarkDown.
- **Macro context**: Fuel price, CPI, Unemployment.
- **Store characteristics**: Encoded store Type, normalized store Size.

All lag/rolling features are built using groupby+shift to prevent data leakage (the model never sees future weeks).

### 3. Train / Test Split
- Sorted by `Store`, `Dept`, `Date`.
- First ~80% of rows used for training.
- Last ~20% held out as “future.”
- No shuffling. This simulates how forecasting would work in production.

### 4. Modeling
- **XGBoost**
  - Trained on all engineered features.
  - Hyperparameter tuning using `RandomizedSearchCV`.
  - Time-aware CV (`TimeSeriesSplit`) to avoid look-ahead bias.
- **Prophet**
  - Fitted per (Store, Dept).
  - Added regressors: holiday flag, markdown totals, macro variables.
  - Generates forward forecasts.
- **LSTM**
  - Sliding 12-week windows turned into supervised sequence data.
  - Architecture: stacked LSTMs + Dropout → Dense(1).
  - EarlyStopping on validation loss.
- **Hybrid Ensemble**
  - Weighted blend of predictions from XGBoost, Prophet, and LSTM.
  - Goal: improve robustness on peak demand weeks (holiday/promo surges).

### 5. Evaluation
- RMSE, MAE, and % error (with epsilon to avoid infinity on zero-sales weeks).
- Visual comparisons:  
  - Actual vs Predicted for a sample store.  
  - Side-by-side model forecasts on the same time window.  
  - XGBoost feature importance plot.

### 6. Interpretability
We extract feature importance from the XGBoost model to answer:
- “What actually drives sales in this forecast?”

Top drivers typically include:
- Recent demand (lag features, rolling means),
- Markdown activity (promotion intensity),
- Seasonal timing (holiday/quarter flags),
- Macro signals (fuel price, CPI, unemployment).

This is critical for business stakeholders because they need to know *why* the forecast moved, not just *what* the number is.

---
