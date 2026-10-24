# Residential-Energy-Forecasting-LSTM-XGBoost-hybrid-model

Day-ahead forecasting of residential electricity load from smart meter data and weather. Buses are grouped by load profile with Ward hierarchical clustering, then many models are trained per cluster: an attention LSTM, XGBoost, an LSTM-XGBoost hybrid, linear and ridge regressions, and an ensemble. Everything is scored against naive baselines, mainly with walk-forward (rolling-origin) evaluation, and tested against imperfect weather forecasts.

## 📁 File Structure

- `IowaState.ipynb`: the full pipeline (data loading, weather, clustering, per-cluster models, baselines, weather features, walk-forward evaluation, transfer learning), saved with its outputs.
- `weather_ames_2017.csv`: cached daily weather for Ames, Iowa (downloaded automatically if missing).
- `ECE910-Project-Report.pdf`: project report.

## 📊 Data

**Load.** Iowa State 240-bus smart meter test system (released 06/17/2019): hourly energy consumption (kWh) at each primary node for January to December 2017, split over three feeders (A: 17 nodes, B: 60, C: 162). The data is not in this repository. Download `Smart Meter Data.xlsx` and set `DATA_DIR` in the first code cell of the notebook.

Preprocessing: all-zero buses are dropped (193 of 239 remain), readings are snapped to the hour (Excel stores times with float drift), the summer window May 1 to Sep 1 is kept, and the hourly data is summed to daily totals (124 days).

**Weather.** Daily mean, max and min temperature, relative humidity, dew point and cooling degree days (CDD, base 18.3 °C) from the Open-Meteo historical archive (ERA5 reanalysis), for Ames, Iowa. The system's location is not stated in the data, so Ames is an assumption; the load tracks Ames temperature closely (correlation with CDD about 0.89 for the three largest clusters), which supports it.

## ⚙️ Method

1. **Clustering.** Each bus's daily load profile is scaled to 0 to 5 and clustered with Ward linkage on the raw profiles, cut into 5 clusters (17, 20, 65, 62 and 29 buses). The clusters mix buses from all three feeders, so they reflect load shape rather than network topology.
2. **Target.** Each cluster's models forecast the cluster's mean daily load, one day ahead.
3. **Features.** Past load: the previous 7 days, a 3-day rolling mean and standard deviation (computed on shifted data so the target day is never included), and day of week. Weather models add temperature, humidity and CDD.
4. **Weather settings.** *Yesterday's weather* is available at forecast time (the honest setting). *Target-day weather* uses the forecast day's observed weather, which assumes a perfect weather forecast. The robustness test (below) replaces "perfect" with a forecast containing error.
5. **Models.** Naive baselines (persistence, same weekday last week, 3-day mean); linear regressions (weekday only; CDD + weekday; CDD + weekday + yesterday's load); a ridge regression on all features plus weather; XGBoost with and without weather; an ensemble averaging the linear model and XGBoost; an attention LSTM; and a hybrid that stacks XGBoost on out-of-fold LSTM predictions.
6. **Evaluation.**
   - *Single split:* chronological 70 / 15 / 15 (81 / 18 / 18 days; test Aug 15 to Sep 1), with scalers fitted on training days only.
   - *Walk-forward (main result):* models are re-fitted weekly on an expanding window (first window 60 days) and forecast the next 7 days one day at a time, pooling 57 test days over 9 re-fits. The stacked hybrid is not repeated here (its out-of-fold stage would multiply the cost); XGBoost uses fixed settings and the LSTM early-stops on the newest 15% of each window.
   - *Forecast-error robustness:* models that use target-day temperature are trained on observed weather but scored with normally distributed noise (σ = 1, 2, 3 °C) added to that day's temperature, 10 draws each.
7. **Transfer learning.** For every cluster as target, the most similar cluster (centroid and PCA distance on training days only) is the source. Zero-shot, from-scratch and fine-tuned models are compared on the single split.

## 📈 Results

### Walk-forward (57 test days per cluster), MAE on scaled load, lower is better

| Cluster | Buses | Best naive | Linear: CDD + weekday + yesterday's load | Ridge: all features + weather | XGBoost + weather | XGBoost only | LSTM only | Ensemble: Linear + XGBoost |
|---|---|---|---|---|---|---|---|---|
| 1 | 17 | 0.095 | 0.046 | 0.050 | 0.054 | 0.068 | 0.165 | **0.045** |
| 2 | 20 | 0.137 | 0.091 | 0.090 | 0.090 | 0.114 | 0.158 | **0.086** |
| 3 | 62 | 0.114 | 0.081 | 0.087 | 0.095 | 0.129 | 0.118 | **0.081** |
| 4 | 65 | 0.121 | **0.088** | 0.092 | 0.105 | 0.138 | 0.129 | 0.092 |
| 5 | 29 | 0.120 | **0.092** | 0.096 | 0.109 | 0.149 | 0.125 | 0.095 |

(Weather models use the target day's observed weather in this table.) In real units, with perfect weather the best model's error is 0.19 to 0.33 kWh per bus per day (7 to 16% of the cluster's mean load), against 0.38 to 0.45 kWh (15 to 21%) for the best naive forecast: better by 52, 37, 29, 27 and 23% for clusters 1 to 5.

### How much of the gain survives an imperfect temperature forecast

Percent better than the best naive forecast, by size of the temperature-forecast error (negative means worse than naive):

| Cluster | Model | σ = 0 °C | σ = 1 °C | σ = 2 °C | σ = 3 °C |
|---|---|---|---|---|---|
| 3 | Linear + yesterday's load | 28.8 | 22.2 | 10.1 | 0.6 |
| 3 | Ridge | 23.3 | 18.9 | 11.5 | 1.2 |
| 3 | Ensemble | 29.4 | 20.6 | 7.7 | -2.2 |
| 4 | Linear + yesterday's load | 27.4 | 18.5 | 7.9 | -13.3 |
| 4 | Ridge | 24.3 | 18.4 | 8.5 | -6.3 |
| 4 | Ensemble | 24.3 | 15.9 | 3.7 | -14.3 |
| 5 | Linear + yesterday's load | 23.0 | 14.9 | 4.4 | -13.3 |
| 5 | Ridge | 19.9 | 14.3 | 6.3 | -8.0 |
| 5 | Ensemble | 20.8 | 12.0 | 0.7 | -14.2 |

### What the results show

- **A simple weather-aware linear model is the most reliable forecaster.** Linear regression on CDD, weekday and yesterday's load is best or within 0.005 MAE of best on every cluster. Averaging it with XGBoost helps slightly on clusters 1 to 3. The LSTM is worse than the best naive forecast on all five clusters in walk-forward.
- **The earlier single-split win for the LSTM on clusters 3 to 5 did not hold up.** On the one 18-day window the LSTM looked best there; across 9 re-fits it loses to naive. That window result was luck, which is why walk-forward is the main evaluation.
- **The weather gain depends heavily on forecast quality.** With a perfect temperature forecast the large clusters gain 23 to 29%; with 1 °C of forecast error the gain is 12 to 22%, with 2 °C only about 1 to 12%, and with 3 °C the models are no better than naive and often worse. The model that uses weather without yesterday's load (CDD + weekday alone) is the most fragile and is worse than naive at σ ≥ 1 °C on clusters 3 to 5. The ridge regression degrades most gracefully at large errors.
- **Clusters 1 and 2 are driven by the weekly cycle** (weekend load is far below weekday load), so the linear, ridge and ensemble models keep roughly 22 to 31% gains over naive even at σ = 3 °C.
- **Transfer learning does not help** (single split): fine-tuning never improved the LSTM, and the hybrid changed by only 0.001 to 0.004 MAE on clusters 4 and 5.
- **The stacked hybrid** was best on one cluster on the single split and was not re-tested in walk-forward.

Earlier versions of this notebook reported sMAPE of 6 to 17%. That came from target leakage, a shuffled split and features built from other buses' same-day load.

## 🚀 How to Run

1. Clone the repository and open the folder.
2. Create an environment and install the dependencies (tested with Python 3.12, pandas 3.1, scikit-learn 1.9, XGBoost 3.4, TensorFlow 2.21):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install pandas openpyxl numpy scipy scikit-learn xgboost tensorflow matplotlib jupyter
   ```
3. Download the smart meter data and set `DATA_DIR` in the notebook's first code cell. Weather is cached in `weather_ames_2017.csv`; if that file is missing, the notebook downloads it (internet needed once).
4. Run `IowaState.ipynb` top to bottom (this takes a while on CPU because of the per-fold LSTM fits; seeds are fixed and rerunning gave identical tables).

## ⚠️ Limitations and next steps

- Weather is ERA5 reanalysis for Ames, not a real day-ahead forecast archive, and the system's true location is unknown. Evaluating against archived forecasts for the right location would replace the noise assumption.
- The noise test assumes independent, zero-mean errors of the same size on mean and max temperature; real forecast errors are correlated across days and vary with the weather.
- One summer of data (2017) means one weather regime; results may not carry over to other years.
- The stacked hybrid and transfer learning are only evaluated on the single split.
- Choose the number of clusters with silhouette or elbow analysis instead of fixing it at 5.
