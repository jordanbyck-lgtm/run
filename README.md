# run: per-run fitness estimation from COROS data

This repo estimates fitness for every run from COROS FIT files. Each estimate combines heart rate, grade-adjusted pace, real weather (NOAA), cardiac drift, interval reps and race efforts. A single run is noisy, so read the trend.

```
pip install fitdecode pandas numpy matplotlib tabulate
# 1. Save COROS MCP `querySportRecords` output (runs) to data/activities_raw.txt
COROS_USER_ID=<id> python run_fitness.py fetch      # FIT files + NOAA GHCNh hourly weather (cached in data/)
python run_fitness.py analyze --hrmax 197 --rhr 45
```
Outputs: `out/runs.csv` (one row per run), `out/fitness.png`, `out/summary.md`.

## Method
1. **Cleaning.** The script resamples each run to 1 Hz. It drops HR spikes, cadence-lock stretches (HR jumps onto the cadence line) and the first 6 minutes, which cover optical-HR lag and warm-up.
2. **Grade-adjusted speed.** Uses the Minetti et al. (2002) energy-cost-vs-grade polynomial.
3. **Settled points.** Only points after ≥2 min at a constant grade-adjusted pace count, with HR lagged 20 s. This keeps HR close to steady state.
4. **Per-run VO2max (`vo2_hr`).** The O2 cost of the speed comes from the ACSM equation (net 0.2 ml/kg/m, about 215–220 ml/kg/km gross at 4:00–6:00/km, which matches trained-runner norms). Swain's %HRR ≈ %VO2R then gives VO2max = 3.5 + (VO2−3.5)/%HRR. The median uses 50–95 %HRR points from minutes 6–40, before drift sets in. `vo2_hr_daniels` uses the Daniels/Gilbert cost curve instead. That curve assumes elite economy at slow speeds (~170 ml/kg/km), so it reads about 10–13 points lower.
5. **HR-vs-speed regression (`vo2_regr`).** Fits HR against speed over settled 30 s bins and extrapolates to HRmax, Firstbeat-style. This only runs with a wide speed spread (r > 0.7). It is averaged in only when it agrees with `vo2_hr` within 5.
6. **Heat.** Uses NOAA GHCNh air temperature and dew point at the run midpoint, from the nearest station within 60 km. Each run's estimate minus the 60-day rolling median is regressed on the excess over 12 °C air and over 10 °C dew. That effect is fit from your own data and then removed (`vo2_adj`, `ef_adj`).
7. **Fitness index (`fitness_vo2`).** A 42-day exponentially weighted mean of `vo2_adj`. It excludes treadmill runs, altitude above 1000 m, runs with too little settled data and runs without weather.
8. **Other per-run metrics.**
   - Efficiency factor (grade-adjusted m/min per beat).
   - Pa:HR decoupling (1st vs 2nd half, runs ≥40 min).
   - Drift in bpm/hour (HR ~ speed + time).
   - Interval reps: count, pace, peak %HRR, HR rise per m/s of speed step, 60 s HR recovery.
9. **Performance anchor (`perf_vdot`).** The fastest 1600 m to marathon segment in each run, kept only if average HR was race-like (≥95% HRmax for 1600 m down to ≥82% for M). It is converted to Daniels VDOT. Calibration is the median of race VDOT divided by the HR index on the same date.

## Caveats
- HR-based VO2max assumes average trained economy. Real economy varies ±10%, which is why a race-calibrated figure is reported alongside it.
- Wrist HR adds noise. Firstbeat-style estimates have a mean absolute error of about 3–5 ml/kg/min with a chest strap.
- Only two race-like efforts exist, so the calibration rests on n=2. They agree with each other: 0.931 and 0.928.
- The FIT URLs are unauthenticated: `s3.coros.com/fit/<userId>/<labelId>.fit`. Keep the user ID and `data/` out of public repos.

## Sources
- Aerobic decoupling (TrainingPeaks): https://www.trainingpeaks.com/coach-blog/aerobic-endurance-and-decoupling/
- Firstbeat VO2max white paper: https://www.firstbeat.com/wp-content/uploads/2017/06/white_paper_VO2max_30.6.2017.pdf
- Swain, %HRR vs %VO2R: https://www.researchgate.net/publication/51300066_Relationship_between_heart_rate_reserve_and_VO2_reserve_in_treadmill_exercise
- ACSM equation overestimates VO2max in athletes by ~14.6%: https://www.hippokratia.gr/images/PDF/17-2/Hippokratia_2_2013_136.pdf
- Running economy ranges: https://en.wikipedia.org/wiki/Running_economy , https://runnersconnect.net/running-economy/
- Daniels/Gilbert VDOT equations: https://rundida.com/tools/vdot-calculator/
- Minetti 2002 grade cost: https://www.researchgate.net/publication/11202969_Energy_Cost_of_Walking_and_Running_at_Extreme_Uphill_and_Downhill_Slopes
- Cadence lock: https://runningwritings.com/2021/05/cadence-lock-why-gps-watches-have-hard.html
- Heat and HR: https://marathonhandbook.com/heat-and-heart-rate/ , https://pubmed.ncbi.nlm.nih.gov/15692320/
- WHOOP VO2max claims and independent checks: https://www.wareable.com/wearable-tech/whoop-vo2-max-estimate-accuracy-rollout , https://boxlifemagazine.com/is-your-vo2-max-accurate/
- NOAA GHCNh hourly: https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly
