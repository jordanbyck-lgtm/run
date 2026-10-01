#!/usr/bin/env python3
"""Quick-and-dirty per-run fitness estimation from COROS FIT files.

Usage:
  python run_fitness.py fetch     # parse data/activities_raw.txt (COROS MCP querySportRecords output), download FITs
  python run_fitness.py analyze   # per-run metrics -> out/runs.csv, out/fitness.png, out/summary.md

Methods (see README.md for sources):
  - VO2 cost of running from ACSM equation (grade-adjusted), %HRR ~ %VO2R (Swain) -> per-point VO2max estimate
  - HR-vs-speed regression across steady segments (Firstbeat-style), extrapolated to HRmax
  - Efficiency factor (grade-adjusted speed / HR), Pa:HR decoupling (1st vs 2nd half)
  - Cardiac drift in bpm/hour controlling for speed (HR ~ speed + time)
  - Interval detection: work reps, HR response per speed step, 60s HR recovery
  - Temperature: watch session avg temp; effect estimated from the data and removed
"""
import argparse, json, os, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA, FIT, OUT = ROOT / "data", ROOT / "data" / "fit", ROOT / "out"
FIT_URL = "https://s3.coros.com/fit/{user}/{label}.fit"


# ---------------------------------------------------------------- fetch
def parse_activity_list(path):
    s = path.read_text()
    if s.startswith('"'):
        s = json.loads(s)
    rows = []
    for blk in re.split(r"\n\d+\. ", s)[1:]:
        g = lambda p: (m.group(1) if (m := re.search(p, blk)) else None)
        rows.append(dict(
            date=g(r"— (\d{4}-\d{2}-\d{2})"), name=g(r"^(.*?) —"), location=g(r"Location: (.*)"),
            start_ts=int(g(r"startTimestamp=(\d+)")), label=g(r"LabelId: (\d+)"), sport=int(g(r"SportType: (\d+)")),
            lat=float(g(r"Coordinates: ([-\d.]+)") or "nan"), lon=float(g(r"Coordinates: [-\d.]+, ([-\d.]+)") or "nan"),
        ))
    return pd.DataFrame(rows)


def fetch(args):
    import urllib.request
    acts = parse_activity_list(DATA / "activities_raw.txt")
    acts.to_csv(DATA / "activities.csv", index=False)
    FIT.mkdir(parents=True, exist_ok=True)
    for lab in acts.label:
        p = FIT / f"{lab}.fit"
        if p.exists() and p.stat().st_size > 0:
            continue
        try:
            urllib.request.urlretrieve(FIT_URL.format(user=args.user, label=lab), p)
        except Exception as e:
            print("fail", lab, e)
    print(f"{len(acts)} activities, {len(list(FIT.glob('*.fit')))} FIT files")


# ---------------------------------------------------------------- per-run
def load_fit(path):
    import fitdecode
    recs, sess, laps = [], {}, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fitdecode.FitReader(str(path)) as r:
            for f in r:
                if not isinstance(f, fitdecode.FitDataMessage):
                    continue
                d = {x.name: x.value for x in f.fields}
                if f.name == "record":
                    recs.append({k: d.get(k) for k in ("timestamp", "heart_rate", "enhanced_speed", "speed",
                                                         "distance", "enhanced_altitude", "altitude", "power")})
                elif f.name == "session":
                    sess = d
                elif f.name == "lap":
                    laps.append(d)
    df = pd.DataFrame(recs)
    if df.empty:
        return df, sess
    df["speed"] = pd.to_numeric(df.enhanced_speed.fillna(df.speed), errors="coerce")
    df["alt"] = pd.to_numeric(df.enhanced_altitude.fillna(df.altitude), errors="coerce")
    df["hr"] = pd.to_numeric(df.heart_rate, errors="coerce")
    df["t"] = (pd.to_datetime(df.timestamp) - pd.to_datetime(df.timestamp.iloc[0])).dt.total_seconds()
    df = df[["t", "hr", "speed", "distance", "alt", "power"]].apply(pd.to_numeric, errors="coerce").astype(float)
    # resample to 1 Hz on elapsed time, mark pauses (gaps) as moving=False
    df = df.drop_duplicates("t").set_index("t").reindex(np.arange(0, df.t.max() + 1)).interpolate(limit=5)
    df.index.name = "t"
    return df.reset_index(), sess


def acsm_vo2(v_ms, grade):
    """ACSM running VO2 (ml/kg/min). Downhill: half credit of the vertical term (crude)."""
    v = v_ms * 60
    g = np.clip(grade, -0.10, 0.15)
    g = np.where(g < 0, g * 0.5, g)
    return 0.2 * v + 0.9 * v * g + 3.5


def analyze_run(df, sess, hrmax, rhr, indoor):
    out = {}
    if df.empty or df.hr.notna().sum() < 300:
        return None
    df = df.copy()
    df["moving"] = df.speed > 1.5  # > ~11 min/km
    # 30 s smoothing; grade from 30 s altitude/distance diff
    sp = df.speed.where(df.moving).rolling(30, center=True, min_periods=20).mean()
    if indoor:
        g = pd.Series(0.0, index=df.index)
    else:
        dd = df.distance.diff(30)
        g = (df.alt.diff(30) / dd).where(dd > 30).rolling(30, center=True, min_periods=10).mean().fillna(0)
    # HR lags workload: compare HR now with workload ~20 s earlier
    lag = 20
    df["v"] = sp.shift(lag)
    df["g"] = g.shift(lag)
    df["vo2"] = acsm_vo2(df.v, df.g)
    df["gap"] = (df.vo2 - 3.5) / 0.2 / 60  # grade-adjusted speed m/s
    df["hrs"] = df.hr.rolling(15, center=True, min_periods=8).mean()
    df["steady"] = (sp.rolling(60, center=True).std() / sp.rolling(60, center=True).mean() < 0.06).shift(lag, fill_value=False)
    mov_t = df.moving.cumsum()  # moving seconds
    df["mt"] = mov_t
    valid = df.moving & df.hrs.notna() & df.v.notna() & (df.hrs > rhr + 20) & (df.hrs < hrmax + 3)

    dur_min = df.moving.sum() / 60
    out.update(moving_min=round(dur_min, 1), dist_km=round(df.distance.max() / 1000, 2),
               avg_hr=round(df.hr[df.moving].mean(), 1), max_hr=df.hr.max(),
               avg_pace=_pace(df.speed[df.moving].mean()), temp_c=sess.get("avg_temperature"),
               ascent_m=sess.get("total_ascent"))

    # --- steady-state points after warmup (6 min)
    pts = df[valid & df.steady & (df.mt > 360)]
    out["n_steady_s"] = len(pts)
    if len(pts) < 120:
        return out
    hrr = (pts.hrs - rhr) / (hrmax - rhr)
    # Swain: %HRR ~ %VO2R -> VO2max = 3.5 + (VO2 - 3.5)/%HRR ; only reasonably loaded points
    m = (hrr > 0.45) & (hrr < 0.97)
    if m.sum() >= 60:
        est = 3.5 + (pts.vo2[m] - 3.5) / hrr[m]
        out["vo2_swain"] = round(est.median(), 1)
    # efficiency factor: grade-adjusted m/min per beat
    out["ef"] = round((pts.gap * 60 / pts.hrs).median(), 3)

    # --- HR ~ speed regression (Firstbeat-style) over 30 s bins; needs speed spread (intervals/progressions help)
    b = pts.groupby((pts.t // 30)).agg(gap=("gap", "mean"), hr=("hrs", "mean"))
    if len(b) >= 8 and b.gap.std() > 0.30:
        slope, icpt = np.polyfit(b.gap, b.hr, 1)
        r = np.corrcoef(b.gap, b.hr)[0, 1]
        out.update(hr_per_ms=round(slope, 1), hr_speed_r=round(r, 2))
        if r > 0.6 and slope > 5:
            v_max = (hrmax - icpt) / slope
            out["vvo2max_pace"] = _pace(v_max)
            out["vo2_regr"] = round(acsm_vo2(v_max, 0.0), 1)

    # --- cardiac drift: HR ~ a + b*gap + c*time  (after 10 min) -> c in bpm/hour
    d = df[valid & (df.mt > 600)]
    if dur_min >= 25 and len(d) > 600:
        X = np.column_stack([np.ones(len(d)), d.gap, d.mt / 3600])
        coef, *_ = np.linalg.lstsq(X, d.hrs, rcond=None)
        out["drift_bpm_per_h"] = round(coef[2], 1)
        # Pa:HR decoupling, first vs second half of moving time (TrainingPeaks definition)
        h = d.mt.median()
        e1 = (d.gap[d.mt <= h] / d.hrs[d.mt <= h]).mean()
        e2 = (d.gap[d.mt > h] / d.hrs[d.mt > h]).mean()
        out["decoupling_pct"] = round((e1 - e2) / e1 * 100, 1)
        cv = d.gap.std() / d.gap.mean()
        out["steady_run"] = bool(cv < 0.10)

    # --- intervals: work reps = 30 s speed > 1.12x run median for >= 45 s
    med = sp[df.moving].median()
    work = (sp > med * 1.12).fillna(False)
    grp = (work != work.shift()).cumsum()
    reps = []
    for _, seg in df[work].groupby(grp[work]):
        if len(seg) < 45:
            continue
        t0, t1 = seg.t.iloc[0], seg.t.iloc[-1]
        pre = df.hrs[(df.t >= t0 - 30) & (df.t < t0)].mean()
        peak = df.hrs[(df.t >= t0) & (df.t <= t1 + 15)].max()
        rec = df.hr[(df.t >= t1 + 55) & (df.t <= t1 + 65)].mean()
        reps.append(dict(dur=len(seg), v=seg.speed.mean(), dhr=peak - pre, hrr60=peak - rec,
                         dv=seg.speed.mean() - df.speed[(df.t >= t0 - 60) & (df.t < t0)].mean()))
    if reps:
        R = pd.DataFrame(reps)
        out.update(n_reps=len(R), rep_pace=_pace(R.v.mean()), rep_peak_hr_rise=round(R.dhr.mean(), 1),
                   hr_rec_60s=round(R.hrr60.mean(), 1),
                   hr_rise_per_ms=round((R.dhr / R.dv.clip(lower=0.2)).median(), 1))
    # progression: positive pace trend across the run (speed vs time slope, m/s per 10 min)
    vv = df.v[df.moving].dropna()
    if len(vv) > 600:
        out["progression"] = round(np.polyfit(np.arange(len(vv)) / 600, vv, 1)[0], 3)
    return out


def _pace(v):
    if not v or not np.isfinite(v) or v <= 0:
        return None
    s = 1000 / v
    return f"{int(s // 60)}:{int(s % 60):02d}"


# ---------------------------------------------------------------- across runs
def analyze(args):
    acts = pd.read_csv(DATA / "activities.csv", dtype={"label": str})
    raw = {}
    for a in acts.itertuples():
        p = FIT / f"{a.label}.fit"
        if p.exists():
            raw[a.label] = load_fit(p)
    hrmax = args.hrmax or float(np.percentile([df.hr.max() for df, _ in raw.values() if not df.empty], 98))
    rhr = args.rhr
    print(f"HRmax={hrmax:.0f} RHR={rhr}")

    rows = []
    for a in acts.itertuples():
        if a.label not in raw:
            continue
        df, sess = raw[a.label]
        r = analyze_run(df, sess, hrmax, rhr, indoor=a.sport == 101)
        if r is None:
            continue
        rows.append(dict(date=a.date, label=a.label, sport=a.sport, location=a.location, **r))
    R = pd.DataFrame(rows).sort_values("date")
    R["date"] = pd.to_datetime(R.date)

    # combined per-run estimate: Swain point estimate, blended with regression when it is credible
    R["vo2_run"] = R.vo2_swain
    both = R.vo2_regr.notna() & R.vo2_swain.notna() & ((R.vo2_regr - R.vo2_swain).abs() < 8)
    R.loc[both, "vo2_run"] = (R.vo2_swain[both] + R.vo2_regr[both]) / 2
    # quality weight: longer steady data, outdoor, low altitude
    R["w"] = np.clip(R.n_steady_s / 1800, 0.2, 1.0) * np.where(R.sport == 101, 0.5, 1.0)

    # temperature effect, estimated from data: residual vs 60-day rolling median ~ temp
    k = 0.0
    ok = R.vo2_run.notna() & R.temp_c.notna()
    if ok.sum() > 30:
        s = R[ok].set_index("date").vo2_run
        trend = s.rolling("60D", center=True, min_periods=5).median()
        resid = (s - trend).values
        temp = R[ok].temp_c.values
        f = np.isfinite(resid)
        k = float(np.polyfit(temp[f], resid[f], 1)[0])
    R["temp_coef"] = round(k, 3)
    tref = args.tref
    R["vo2_adj"] = R.vo2_run - k * (R.temp_c - tref)
    kef = 0.0
    ok = R.ef.notna() & R.temp_c.notna()
    if ok.sum() > 30:
        s = R[ok].set_index("date").ef
        resid = (s - s.rolling("60D", center=True, min_periods=5).median()).values
        f = np.isfinite(resid)
        kef = float(np.polyfit(R[ok].temp_c.values[f], resid[f], 1)[0])
    R["ef_adj"] = R.ef - kef * (R.temp_c - tref)

    # fitness index: time-weighted EWMA (42 d) of temp-adjusted per-run VO2 estimates
    fit, num, den, last = [], 0.0, 0.0, None
    for row in R.itertuples():
        if last is not None:
            decay = np.exp(-(row.date - last).days / 42)
            num, den = num * decay, den * decay
        if pd.notna(row.vo2_adj):
            num, den = num + row.w * row.vo2_adj, den + row.w
        last = row.date
        fit.append(num / den if den else np.nan)
    R["fitness_vo2"] = np.round(fit, 1)

    OUT.mkdir(exist_ok=True)
    R.to_csv(OUT / "runs.csv", index=False)
    plot(R)
    summary(R, hrmax, rhr, k, kef, tref)
    print(f"wrote {OUT}/runs.csv, fitness.png, summary.md  ({len(R)} runs)")


def plot(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    ax[0].scatter(R.date, R.vo2_adj, s=10 + 30 * R.w, alpha=.45, color="#4C78A8", label="per-run est (temp-adj)")
    ax[0].plot(R.date, R.fitness_vo2, color="#E45756", lw=2, label="fitness (42d EWMA)")
    ax[0].set_ylabel("VO2max est"); ax[0].legend(loc="lower left")
    ax[1].scatter(R.date, R.ef_adj, s=12, alpha=.5, color="#4C78A8")
    ax[1].plot(R.date, R.set_index("date").ef_adj.rolling("28D").median().values, color="#E45756")
    ax[1].set_ylabel("EF (GAP m/min / bpm)")
    st = R[R.steady_run == True]
    ax[2].scatter(st.date, st.decoupling_pct, s=12, alpha=.6, color="#4C78A8", label="decoupling %")
    ax[2].scatter(R.date, R.drift_bpm_per_h, s=10, alpha=.4, color="#F58518", label="drift bpm/h")
    ax[2].axhline(5, ls=":", c="grey"); ax[2].legend(loc="upper left"); ax[2].set_ylabel("drift")
    ax[3].scatter(R.date, R.temp_c, s=10, color="#F58518"); ax[3].set_ylabel("watch temp °C")
    fig.tight_layout(); fig.savefig(OUT / "fitness.png", dpi=110)


def summary(R, hrmax, rhr, k, kef, tref):
    m = R.set_index("date").resample("MS").agg(
        runs=("label", "count"), km=("dist_km", "sum"), vo2_adj=("vo2_adj", "median"), fitness=("fitness_vo2", "last"),
        ef_adj=("ef_adj", "median"), decoup=("decoupling_pct", "median"), drift=("drift_bpm_per_h", "median"),
        temp=("temp_c", "mean"), hr_rec_60s=("hr_rec_60s", "median"))
    m.index = m.index.strftime("%Y-%m")
    lines = [
        "# Run fitness summary", "",
        f"- Runs analyzed: {len(R)} ({R.date.min():%Y-%m-%d} → {R.date.max():%Y-%m-%d})",
        f"- HRmax used: {hrmax:.0f} (98th pct of per-run max) · RHR: {rhr} (assumed; COROS returned none)",
        f"- Temperature effect (from data): {k:+.3f} VO2 units/°C, {kef:+.4f} EF/°C; normalized to {tref}°C",
        f"- Latest fitness index (42d EWMA VO2max est): **{R.fitness_vo2.iloc[-1]}**",
        "", "## Monthly", "", m.round(2).to_markdown(), "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "analyze"])
    ap.add_argument("--user", default=os.environ.get("COROS_USER_ID"), help="COROS user id in FIT URLs (or $COROS_USER_ID)")
    ap.add_argument("--hrmax", type=float)
    ap.add_argument("--rhr", type=float, default=50)
    ap.add_argument("--tref", type=float, default=15)
    a = ap.parse_args()
    {"fetch": fetch, "analyze": analyze}[a.cmd](a)
