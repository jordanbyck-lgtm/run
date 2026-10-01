# run — per-run fitness estimation from COROS data

A rough per-run fitness estimate for every run, built from COROS FIT files. It uses HR, grade-adjusted pace, temperature, cardiac drift and interval reps, not just average pace.

```
pip install fitdecode pandas numpy matplotlib tabulate
# 1. Save COROS MCP `querySportRecords` output (runs, sport 100-103) to data/activities_raw.txt
# 2. Download FITs and analyze
COROS_USER_ID=<id> python run_fitness.py fetch
python run_fitness.py analyze --rhr 50     # optional --hrmax, --tref
```
Outputs: `out/runs.csv` (one row per run), `out/fitness.png`, `out/summary.md`.

## What it computes per run
| metric | method |
|---|---|
| `vo2_swain` | Grade-adjusted VO2 cost (ACSM running eq.). %HRR is taken to equal %VO2R (Swain), so VO2max = 3.5 + (VO2−3.5)/%HRR. Median over steady 1 Hz points after the 6-min warm-up, with HR lagged 20 s behind speed. |
| `vo2_regr` | HR-vs-speed regression over 30 s bins, extrapolated to HRmax (Firstbeat-style). Only computed when the speed spread is wide enough, e.g. intervals or progressions. |
| `vo2_run` / `vo2_adj` | Swain estimate, averaged with the regression estimate when the two agree within 8. `vo2_adj` is temperature-normalized to 15 °C. |
| `fitness_vo2` | 42-day exponentially weighted average of `vo2_adj`, weighted by how much steady data each run has. Treadmill runs get half weight. |
| `ef` / `ef_adj` | Efficiency factor: grade-adjusted m/min per beat (Friel/TrainingPeaks). |
| `decoupling_pct` | Pa:HR, 1st vs 2nd half of moving time after 10 min. Only meaningful when `steady_run`. |
| `drift_bpm_per_h` | Time coefficient of HR ~ speed + time, so pace changes are controlled for. |
| `n_reps`, `rep_*`, `hr_rise_per_ms`, `hr_rec_60s` | Detected work reps (30 s speed >1.12× run median for ≥45 s). Reports HR rise per m/s speed step and the HR drop 60 s after each rep. |
| `progression` | Speed trend across the run (m/s per 10 min). |
| temp | Taken from the watch session's `avg_temperature`. The temperature effect is fit from the data as the residual against a 60-day rolling median, then removed. |

## Caveats (quick and dirty)
- **Watch temperature is skin-biased.** It reads about 20 °C on winter runs, so the fitted heat coefficient is understated. Real weather data (e.g. Open-Meteo archive by start coords and time) would be better. It was rate-limited from this environment.
- **RHR is assumed (50).** COROS returned no resting HR. HRmax is the 98th percentile of per-run max HR.
- **Wrist HR is noisy.** Firstbeat-style estimates have a mean absolute error of about 3–5 ml/kg/min with a chest strap and worse with wrist HR. Read the trend, not single runs.
- Downhill grade gets crude half credit. Rep detection also picks up some hills and surges.
- **FIT URLs are unauthenticated.** They are just `s3.coros.com/fit/<userId>/<labelId>.fit`, so keep the user ID and `data/` out of public repos.

## Sources
- TrainingPeaks, aerobic decoupling / Pa:HR: https://www.trainingpeaks.com/coach-blog/aerobic-endurance-and-decoupling/
- Firstbeat VO2max white paper (HR–speed regression): https://www.firstbeat.com/wp-content/uploads/2017/06/white_paper_VO2max_30.6.2017.pdf
- Swain, %HRR vs %VO2R in treadmill exercise: https://www.researchgate.net/publication/51300066_Relationship_between_heart_rate_reserve_and_VO2_reserve_in_treadmill_exercise
- ACSM running equation in athletes: https://www.hippokratia.gr/images/PDF/17-2/Hippokratia_2_2013_136.pdf
- Heat and pace/HR: https://www.outsideonline.com/running/racing/race-strategy/how-much-does-heat-slow-your-race-pace/ , https://www.polar.com/en/media-room/physiology-behind-hot-weather-running
