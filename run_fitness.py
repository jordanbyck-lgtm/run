#!/usr/bin/env python3
"""Per-run fitness estimation from COROS FIT files (v2).

Usage:
  COROS_USER_ID=<id> python run_fitness.py fetch   # parse data/activities_raw.txt, download FITs + NOAA weather
  python run_fitness.py analyze [--hrmax 197 --rhr 45]

Pipeline per run (see README.md for method notes and sources):
  1. 1 Hz streams; drop wrist-HR artifacts (cadence lock, spikes) and the first 6 min (optical lag, warm-up)
  2. Grade-adjusted speed via Minetti (2002) energy-cost polynomial
  3. "Settled" points: >= 2 min at a constant grade-adjusted pace, so HR is near steady state
  4. VO2 cost of that speed from ACSM (net 0.2 ml/kg/m ~ trained-runner economy at these speeds);
     Daniels/Gilbert kept for VDOT / race equivalents (its slow-speed cost is elite-level, ~170 ml/kg/km)
  5. Swain: %HRR ~= %VO2R  ->  VO2max = 3.5 + (VO2 - 3.5) / %HRR, using minutes 6-40 (pre-drift)
  6. HR-vs-speed regression across settled bins, extrapolated to HRmax (Firstbeat-style), when speed spread allows
  7. Drift (HR ~ speed + time), Pa:HR decoupling, interval reps, best-effort VDOT (performance anchor)
Across runs: heat/dew-point effect fit from data using NOAA GHCNh air temp + dew point, removed; 42-day EWMA index;
calibration of the HR index against HR-confirmed hard efforts (performance VDOT).
"""
import argparse, io, json, os, re, urllib.request, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA, FIT, WX, OUT = ROOT / "data", ROOT / "data" / "fit", ROOT / "data" / "weather", ROOT / "out"
FIT_URL = "https://s3.coros.com/fit/{user}/{label}.fit"
GHCNH = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly"
GHCNH_FILE = GHCNH + "/access/by-year/{y}/psv/GHCNh_{st}_{y}.psv"
GHCNH_STATIONS = GHCNH + "/doc/ghcnh-station-list.txt"
# alternative HR anchors for the sensitivity table (rhr, hrmax)
SCENARIOS = [(45, 197), (50, 197), (45, 193), (45, 200), (40, 197)]


# ================================================================ fetch
def parse_activity_list(path):
    s = path.read_text()
    if s.startswith('"'):
        s = json.loads(s)
    rows = []
    for blk in re.split(r"\n\d+\. ", s)[1:]:
        g = lambda p: (m.group(1) if (m := re.search(p, blk)) else None)
        rows.append(dict(
            date=g(r"— (\d{4}-\d{2}-\d{2})"), location=g(r"Location: (.*)"),
            start_ts=int(g(r"startTimestamp=(\d+)")), end_ts=int(g(r"endTimestamp=(\d+)")),
            label=g(r"LabelId: (\d+)"), sport=int(g(r"SportType: (\d+)")),
            lat=float(g(r"Coordinates: ([-\d.]+)") or "nan"), lon=float(g(r"Coordinates: [-\d.]+, ([-\d.]+)") or "nan"),
        ))
    return pd.DataFrame(rows)


def _haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def load_stations():
    p = DATA / "ghcnh-stations.txt"
    if not p.exists():
        urllib.request.urlretrieve(GHCNH_STATIONS, p)
    rows = [(l[:11], float(l[12:20]), float(l[21:30])) for l in p.read_text().splitlines() if len(l) > 30]
    return pd.DataFrame(rows, columns=["id", "lat", "lon"])


def nearest_stations(lat, lon, st, n=4):
    d = _haversine(lat, lon, st.lat.values, st.lon.values)
    i = np.argsort(d)[:n]
    return [(st.id.iloc[j], round(float(d[j]), 1)) for j in i]


def _wx_file(st, year):
    """Cached slim GHCNh station-year (time, temp, dew). Returns None if unavailable."""
    p = WX / f"{st}_{year}.csv"
    if not p.exists():
        try:
            with urllib.request.urlopen(GHCNH_FILE.format(st=st, y=year), timeout=300) as r:
                w = pd.read_csv(r, sep="|", usecols=["DATE", "temperature", "dew_point_temperature"], low_memory=False)
            w.columns = ["time", "air_c", "dew_c"]
            w.dropna(subset=["air_c"]).to_csv(p, index=False)
        except Exception as e:
            print("wx miss", st, year, type(e).__name__)
            p.write_text("time,air_c,dew_c\n")
    w = pd.read_csv(p)
    return w if len(w) else None


def fetch(args):
    acts = parse_activity_list(DATA / "activities_raw.txt")
    FIT.mkdir(parents=True, exist_ok=True)
    WX.mkdir(parents=True, exist_ok=True)
    for lab in acts.label:
        p = FIT / f"{lab}.fit"
        if not (p.exists() and p.stat().st_size > 0):
            try:
                urllib.request.urlretrieve(FIT_URL.format(user=args.user, label=lab), p)
            except Exception as e:
                print("fit fail", lab, e)
    st = load_stations()
    acts["stations"] = [json.dumps(nearest_stations(a.lat, a.lon, st)) if a.sport != 101 and np.isfinite(a.lat)
                        else None for a in acts.itertuples()]
    acts.to_csv(DATA / "activities.csv", index=False)
    for a in acts.itertuples():
        if isinstance(a.stations, str):
            run_weather(a)  # warms the cache
    print(f"{len(acts)} activities, {len(list(FIT.glob('*.fit')))} FIT files, {len(list(WX.glob('*.csv')))} weather files")


def run_weather(a):
    """Air temp and dew point (°C) nearest the run midpoint, from the closest station (<= 60 km) with an obs within 90 min."""
    if not isinstance(a.stations, str):
        return np.nan, np.nan, None
    mid = pd.Timestamp((a.start_ts + a.end_ts) / 2, unit="s", tz="UTC")
    for st, km in json.loads(a.stations):
        if km > 60:
            break
        w = _wx_file(st, mid.year)
        if w is None:
            continue
        t = pd.to_datetime(w.time, utc=True)
        dt = (t - mid).abs()
        i = dt.idxmin()
        if dt[i] <= pd.Timedelta(minutes=90):
            return w.air_c[i], w.dew_c[i], f"{st}@{km}km"
    return np.nan, np.nan, None


# ================================================================ physiology helpers
def minetti_factor(i):
    """Cost of running at grade i relative to flat (Minetti et al. 2002)."""
    i = np.clip(i, -0.25, 0.25)
    c = 155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3 + 46.3 * i**2 + 19.5 * i + 3.6
    return c / 3.6


def daniels_vo2(v_ms):
    """Daniels/Gilbert oxygen cost (ml/kg/min) of running at v (m/s), trained-runner economy."""
    v = v_ms * 60
    return -4.60 + 0.182258 * v + 0.000104 * v**2


def acsm_vo2(v_ms):
    return 0.2 * v_ms * 60 + 3.5


def daniels_pct(t_min):
    """Fraction of VO2max sustainable for a race of t minutes (Daniels/Gilbert)."""
    return 0.8 + 0.1894393 * np.exp(-0.012778 * t_min) + 0.2989558 * np.exp(-0.1932605 * t_min)


def vdot(dist_m, t_s):
    t = t_s / 60
    return daniels_vo2(dist_m / t_s) / daniels_pct(t)


def vdot_race_time(v, dist_m):
    """Invert VDOT -> race time (s) for a distance by bisection."""
    lo, hi = dist_m / 8, dist_m / 1.5
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if vdot(dist_m, mid) > v else (lo, mid)
    return mid


def _pace(v):
    if v is None or not np.isfinite(v) or v <= 0:
        return None
    s = 1000 / v
    return f"{int(s // 60)}:{int(round(s % 60)):02d}"


def _hms(s):
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


# ================================================================ per-run
def load_fit(path):
    import fitdecode
    recs, sess = [], {}
    keys = ("timestamp", "heart_rate", "enhanced_speed", "speed", "distance", "enhanced_altitude", "altitude",
            "cadence", "fractional_cadence")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fitdecode.FitReader(str(path)) as r:
            for f in r:
                if not isinstance(f, fitdecode.FitDataMessage):
                    continue
                if f.name == "record":
                    d = {x.name: x.value for x in f.fields}
                    recs.append({k: d.get(k) for k in keys})
                elif f.name == "session":
                    sess = {x.name: x.value for x in f.fields}
    df = pd.DataFrame(recs)
    if df.empty:
        return df, sess
    ts = pd.to_datetime(df.timestamp)
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    out = pd.DataFrame({
        "t": (ts - ts.iloc[0]).dt.total_seconds(),
        "hr": num("heart_rate"),
        "speed": num("enhanced_speed").fillna(num("speed")),
        "distance": num("distance"),
        "alt": num("enhanced_altitude").fillna(num("altitude")),
        "spm": 2 * (num("cadence") + num("fractional_cadence").fillna(0)),
    }).astype(float)
    out = out.drop_duplicates("t").set_index("t")
    out = out.reindex(np.arange(0, out.index.max() + 1)).interpolate(limit=5, limit_area="inside")
    out.index.name = "t"
    return out.reset_index(), sess


def analyze_run(df, hrmax, rhr, indoor):
    if df.empty or df.hr.notna().sum() < 600:
        return None
    df = df.copy()
    o = {}
    lag = 20  # HR responds to workload ~20 s later

    # ---- HR cleaning
    hr = df.hr.where((df.hr > 60) & (df.hr < 215))
    med = hr.rolling(31, center=True, min_periods=10).median()
    hr = hr.where((hr - med).abs() < 12)                                   # spikes
    # cadence lock: HR sits on the cadence line AND jumped onto it (a real HR ~= cadence at race effort is kept)
    near = ((hr - df.spm).abs() < 4).rolling(30, center=True, min_periods=15).mean() > 0.8
    lock = pd.Series(False, index=df.index)
    seg = (near != near.shift()).cumsum()
    for _, ix in near[near].groupby(seg[near]).groups.items():
        i0 = ix[0]
        jump = hr.loc[max(i0 - 20, 0):i0 + 10].max() - hr.loc[max(i0 - 40, 0):max(i0 - 15, 0)].median()
        if len(ix) >= 30 and jump >= 12:
            lock.loc[ix] = True
    o["hr_lock_pct"] = round(100 * lock[df.speed > 1.8].mean(), 1)        # cadence lock
    hr = hr.where(~lock)
    df["hrs"] = hr.rolling(15, center=True, min_periods=8).mean()

    # ---- speed, grade, grade-adjusted speed
    df["moving"] = df.speed > 1.8                                          # faster than ~9:15/km
    sp = df.speed.where(df.moving).rolling(30, center=True, min_periods=20).mean()
    if indoor:
        gf = pd.Series(1.0, index=df.index)
    else:
        alt = df.alt.rolling(30, center=True, min_periods=10).mean()
        dd = df.distance.diff(30)
        grade = (alt.diff(30) / dd).where(dd > 40).rolling(30, center=True, min_periods=10).mean().fillna(0)
        gf = pd.Series(minetti_factor(grade.values), index=df.index)
    gap = sp * gf
    df["gap"] = gap.shift(lag)
    df["mt"] = df.moving.cumsum()                                          # moving seconds
    # settled: trailing 120 s of near-constant grade-adjusted speed (then lagged)
    cv = gap.rolling(120, min_periods=100).std() / gap.rolling(120, min_periods=100).mean()
    df["settled"] = (cv < 0.06).shift(lag, fill_value=False)
    df["vo2"] = acsm_vo2(df.gap)
    valid = df.moving & df.hrs.notna() & df.gap.notna() & (df.mt > 360)

    dur = df.moving.sum() / 60
    o.update(moving_min=round(dur, 1), dist_km=round(df.distance.max() / 1000, 2),
             avg_hr=round(df.hr[df.moving].mean(), 1), max_hr=df.hr.max(), avg_pace=_pace(df.speed[df.moving].mean()),
             gap_pace=_pace((df.speed * gf)[df.moving].mean()), median_alt_m=round(df.alt.median(), 0))

    # ---- Swain estimate (pre-drift window)
    pts = df[valid & df.settled & (df.mt <= 2400)]
    o["n_settled_s"] = len(pts)
    if len(pts) >= 180:
        for r_, m_ in SCENARIOS:
            hrr = (pts.hrs - r_) / (m_ - r_)
            k = (hrr > 0.5) & (hrr < 0.95)
            if k.sum() >= 120:
                est = 3.5 + (pts.vo2[k] - 3.5) / hrr[k]
                o[f"vo2_{r_}_{m_}"] = round(est.median(), 2)
        hrr = (pts.hrs - rhr) / (hrmax - rhr)
        o["pct_hrr"] = round(hrr.median(), 3)
        k = (hrr > 0.5) & (hrr < 0.95)
        if k.sum() >= 120:
            o["vo2_hr"] = round((3.5 + (pts.vo2[k] - 3.5) / hrr[k]).median(), 2)
            o["vo2_hr_daniels"] = round((3.5 + (daniels_vo2(pts.gap[k]) - 3.5) / hrr[k]).median(), 2)
        o["ef"] = round((pts.gap * 60 / pts.hrs).median(), 3)          # m/min per beat (grade adjusted)

    # ---- HR-vs-speed regression over settled 30 s bins (whole run; intervals/progressions give spread)
    sb = df[valid & df.settled]
    b = sb.groupby(sb.t // 30).agg(gap=("gap", "mean"), hr=("hrs", "mean"))
    if len(b) >= 10 and b.gap.std() > 0.25:
        slope, icpt = np.polyfit(b.gap, b.hr, 1)
        r = np.corrcoef(b.gap, b.hr)[0, 1]
        o.update(hr_per_ms=round(slope, 1), hr_speed_r=round(r, 2))
        if r > 0.7 and slope > 5:
            o["vo2_regr"] = round(acsm_vo2((hrmax - icpt) / slope), 2)

    # ---- cardiac drift & decoupling (after 10 min, runs >= 40 min, roughly steady)
    d = df[valid & (df.mt > 600)]
    if dur >= 40 and len(d) > 1200:
        X = np.column_stack([np.ones(len(d)), d.gap, d.mt / 3600])
        coef, *_ = np.linalg.lstsq(X, d.hrs, rcond=None)
        o["drift_bpm_per_h"] = round(coef[2], 1)
        h = d.mt.median()
        e1 = (d.gap[d.mt <= h] / d.hrs[d.mt <= h]).mean()
        e2 = (d.gap[d.mt > h] / d.hrs[d.mt > h]).mean()
        o["decoupling_pct"] = round((e1 - e2) / e1 * 100, 1)
        o["steady_run"] = bool(d.gap.std() / d.gap.mean() < 0.10)

    # ---- interval reps: grade-adjusted speed >= 1.15x run median for >= 60 s, a real step up
    g0 = gap[df.moving].median()
    work = (gap > g0 * 1.15).fillna(False)
    grp = (work != work.shift()).cumsum()
    reps = []
    for _, seg in df[work].groupby(grp[work]):
        if len(seg) < 60:
            continue
        t0, t1 = seg.t.iloc[0], seg.t.iloc[-1]
        before = gap[(df.t >= t0 - 90) & (df.t < t0 - 15)].mean()
        after = gap[(df.t > t1 + 10) & (df.t <= t1 + 60)].mean()
        if not (gap[seg.index].mean() - before >= 0.4):
            continue
        pre = df.hrs[(df.t >= t0 - 20) & (df.t < t0)].mean()
        peak = df.hrs[(df.t >= t0) & (df.t <= t1 + 10)].max()
        rec = df.hrs[(df.t >= t1 + 55) & (df.t <= t1 + 65)].mean()
        reps.append(dict(dur=len(seg), v=gap[seg.index].mean(), dv=gap[seg.index].mean() - before,
                         peak_hrr=(peak - rhr) / (hrmax - rhr), dhr=peak - pre,
                         hrr60=(peak - rec) if after < gap[seg.index].mean() - 0.5 else np.nan))
    if reps:
        R = pd.DataFrame(reps)
        o.update(n_reps=len(R), rep_dur_s=int(R.dur.median()), rep_gap_pace=_pace(R.v.mean()),
                 rep_peak_pct_hrr=round(R.peak_hrr.median(), 3),
                 rep_hr_rise_per_ms=round((R.dhr / R.dv).median(), 1), rep_hr_rec_60s=round(R.hrr60.median(), 1))

    # ---- best efforts (performance anchor): fastest 1600 / 3000 / 5000 / 10000 m in the run
    dist = df.distance.ffill()
    tt = df.t.values
    best = None
    for D in (1600, 3000, 5000, 10000, 21097.5, 42195):
        if dist.max() < D:
            continue
        j = np.searchsorted(dist.values, dist.values + D)
        ok = j < len(dist)
        if not ok.any():
            continue
        dts = tt[j[ok]] - tt[ok]
        i = int(np.argmin(dts))
        seg_hr = df.hr.iloc[i:j[ok][i]]
        v = vdot(D, dts[i])
        o[f"best_{D}"] = _hms(dts[i])
        o[f"best_{D}_hrmax_pct"] = round(seg_hr.mean() / hrmax, 3)
        # only count it as a performance if avg HR says it was race-like (typical race %HRmax by duration)
        if seg_hr.mean() / hrmax >= {1600: 0.95, 3000: 0.93, 5000: 0.91, 10000: 0.90, 21097.5: 0.87, 42195: 0.82}[D]:
            if best is None or v > best[0]:
                best = (v, D)
    if best:
        o["perf_vdot"], o["perf_dist"] = round(best[0], 1), best[1]
    return o


# ================================================================ across runs
def analyze(args):
    acts = pd.read_csv(DATA / "activities.csv", dtype={"label": str})
    hrmax, rhr = args.hrmax, args.rhr
    rows = []
    for a in acts.itertuples():
        p = FIT / f"{a.label}.fit"
        if not p.exists():
            continue
        df, _ = load_fit(p)
        r = analyze_run(df, hrmax, rhr, indoor=a.sport == 101)
        if r is None:
            continue
        tmp, dew, st = run_weather(a)
        rows.append(dict(date=a.date, label=a.label, sport=a.sport, location=a.location,
                         air_c=tmp, dew_c=dew, wx_station=st, **r))
    R = pd.DataFrame(rows)
    R["date"] = pd.to_datetime(R.date)
    R = R.sort_values("date").reset_index(drop=True)
    # exclusions from the fitness index
    R["exclude"] = ""
    R.loc[R.sport == 101, "exclude"] += "treadmill;"
    R.loc[R.median_alt_m > 1000, "exclude"] += "altitude;"
    R.loc[R.hr_lock_pct > 20, "exclude"] += "hr_lock;"
    R.loc[R.vo2_hr.isna(), "exclude"] += "too_little_settled_data;"
    R.loc[R.air_c.isna() & (R.sport != 101), "exclude"] += "no_weather;"
    use = R.exclude == ""

    # combined per-run HR estimate: Swain, averaged with regression only when they agree
    R["vo2_run"] = R.vo2_hr
    both = R.vo2_regr.notna() & R.vo2_hr.notna() & ((R.vo2_regr - R.vo2_hr).abs() < 5)
    R.loc[both, "vo2_run"] = (R.vo2_hr[both] + R.vo2_regr[both]) / 2

    # ---- heat model, fit from the data: residual vs 60-day rolling median ~ heat terms
    def heat_terms(t, d):
        return np.column_stack([np.clip(t - 12, 0, None), np.clip(d - 10, 0, None)])

    def fit_heat(col):
        s = R.loc[use].set_index("date")[col]
        trend = s.rolling("60D", center=True, min_periods=6).median()
        resid = (s - trend).values
        X = heat_terms(R.loc[use, "air_c"].values, R.loc[use, "dew_c"].fillna(R.loc[use, "air_c"] - 8).values)
        f = np.isfinite(resid) & np.isfinite(X).all(1)
        X1 = np.column_stack([np.ones(f.sum()), X[f]])
        coef, *_ = np.linalg.lstsq(X1, resid[f], rcond=None)
        return np.minimum(coef[1:], 0)  # heat can only depress the estimate

    kv = fit_heat("vo2_run")
    ke = fit_heat("ef")
    X = heat_terms(R.air_c.fillna(15).values, R.dew_c.fillna(R.air_c - 8).fillna(7).values)
    R["heat_adj_vo2"] = -(X @ kv)
    R["vo2_adj"] = R.vo2_run + R.heat_adj_vo2
    R["ef_adj"] = R.ef - X @ ke

    # ---- fitness index: time-decayed weighted mean (42 d) over eligible runs
    R["w"] = np.clip(R.n_settled_s / 1200, 0.25, 1.0) * (1 - R.hr_lock_pct.fillna(0) / 100)

    def ewma(col):
        out, num, den, last, last_ok = [], 0.0, 0.0, None, None
        for row in R.itertuples():
            if last is not None:
                dec = np.exp(-(row.date - last).days / 42)
                num, den = num * dec, den * dec
            v = getattr(row, col)
            if row.exclude == "" and pd.notna(v):
                num, den, last_ok = num + row.w * v, den + row.w, row.date
            last = row.date
            fresh = last_ok is not None and (row.date - last_ok).days <= 21
            out.append(round(num / den, 2) if den > 0.5 and fresh else np.nan)
        return out

    R["fitness_vo2"] = ewma("vo2_adj")
    # performance anchor: best HR-confirmed hard-effort VDOT in trailing 90 days
    R["perf_vdot_90d"] = R.set_index("date").perf_vdot.rolling("90D").max().values

    # calibration: how the HR index compares with HR-confirmed hard efforts on the same dates
    P = R[R.perf_vdot.notna() & R.fitness_vo2.notna()]
    R.attrs["calib"] = (P.perf_vdot / P.fitness_vo2).median() if len(P) else np.nan
    R.attrs["calib_n"] = len(P)

    # sensitivity: index under other HR anchors (heat model held fixed)
    sens = {}
    last = R.date.max()
    recent = use & (R.date > last - pd.Timedelta(days=42))
    for r_, m_ in SCENARIOS:
        c = f"vo2_{r_}_{m_}"
        if c in R:
            v = (R.loc[recent, c] + R.loc[recent, "heat_adj_vo2"]).dropna()
            if len(v):
                sens[(r_, m_)] = np.average(v, weights=R.loc[v.index, "w"])

    OUT.mkdir(exist_ok=True)
    keep = [c for c in R.columns if not re.match(r"vo2_\d+_\d+", c)]
    R[keep].to_csv(OUT / "runs.csv", index=False)
    plot(R, use)
    summary(R, use, hrmax, rhr, kv, ke, sens)
    print(f"wrote {OUT}/runs.csv, fitness.png, summary.md ({len(R)} runs, {use.sum()} in index)")


def plot(R, use):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, aqua, ink, muted = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": muted,
                         "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted, "font.size": 9,
                         "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fig, ax = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    U = R[use]
    ax[0].scatter(U.date, U.vo2_adj, s=10 + 30 * U.w, alpha=.45, color=blue, label="per-run HR estimate (heat-adjusted)")
    ax[0].plot(R.date, R.fitness_vo2, color=orange, lw=2, label="fitness index (42-day weighted)")
    P = R[R.perf_vdot.notna()]
    ax[0].scatter(P.date, P.perf_vdot, marker="^", s=40, color=aqua, label="hard-effort VDOT (performance)")
    ax[0].set_ylabel("VO2max (ml/kg/min)"); ax[0].legend(loc="lower left", frameon=False)
    ax[0].set_title("VO2max estimate per run", loc="left", color=ink)
    ax[1].scatter(U.date, U.ef_adj, s=12, alpha=.5, color=blue)
    ax[1].plot(U.date, U.set_index("date").ef_adj.rolling("28D").median().values, color=orange, lw=2)
    ax[1].set_ylabel("EF (GAP m/min per beat)"); ax[1].set_title("Efficiency factor, heat-adjusted (28-day median line)", loc="left", color=ink)
    S = R[(R.steady_run == True) & use]
    ax[2].scatter(S.date, S.decoupling_pct, s=14, alpha=.7, color=blue)
    ax[2].axhline(5, ls=":", c=muted); ax[2].set_ylabel("Pa:HR decoupling %")
    ax[2].set_title("Decoupling on steady runs ≥40 min (5% guide line)", loc="left", color=ink)
    D = R[R.drift_bpm_per_h.notna() & use]
    ax[3].scatter(D.date, D.drift_bpm_per_h, s=14, alpha=.7, color=blue); ax[3].axhline(0, c=muted, lw=.8)
    ax[3].set_ylabel("bpm / hour"); ax[3].set_title("Cardiac drift at constant grade-adjusted pace", loc="left", color=ink)
    ax[4].scatter(R.date, R.air_c, s=12, color=orange, label="air temp")
    ax[4].scatter(R.date, R.dew_c, s=12, color=blue, label="dew point")
    ax[4].set_ylabel("°C (NOAA)"); ax[4].legend(loc="lower left", frameon=False)
    ax[4].set_title("Weather at run midpoint", loc="left", color=ink)
    fig.tight_layout(); fig.savefig(OUT / "fitness.png", dpi=110)


def summary(R, use, hrmax, rhr, kv, ke, sens):
    U = R[use]
    m = U.set_index("date").resample("MS").agg(
        runs=("label", "count"), km=("dist_km", "sum"), vo2_adj=("vo2_adj", "median"), index=("fitness_vo2", "last"),
        ef_adj=("ef_adj", "median"), decoup=("decoupling_pct", "median"), drift=("drift_bpm_per_h", "median"),
        air=("air_c", "mean"), dew=("dew_c", "mean"), perf_vdot=("perf_vdot", "max"))
    m.index = m.index.strftime("%Y-%m")
    last = R.iloc[-1]
    rec = U[U.date > R.date.max() - pd.Timedelta(days=42)].vo2_adj
    pv = R.perf_vdot_90d.iloc[-1]
    pr = lambda v: " · ".join(f"{n} {_hms(vdot_race_time(v, d))}" for n, d in
                              (("5k", 5000), ("10k", 10000), ("HM", 21097.5), ("M", 42195)))
    ex = R.exclude.str.strip(";").replace("", "ok").value_counts().to_dict()
    lines = [
        "# Run fitness summary", "",
        f"Runs: {len(R)} ({R.date.min():%Y-%m-%d} → {R.date.max():%Y-%m-%d}), {use.sum()} in index. Exclusions: {ex}",
        f"HRmax {hrmax:.0f}, RHR {rhr:.0f}. Heat model (fit from data, per °C over 12 °C air / 10 °C dew): "
        f"VO2 {kv[0]:+.2f} / {kv[1]:+.2f}, EF {ke[0]:+.4f} / {ke[1]:+.4f}", "",
        "## Current read", "",
        f"- **HR-based fitness index: {last.fitness_vo2:.1f}** (last 42 d per-run IQR {rec.quantile(.25):.1f}–{rec.quantile(.75):.1f})",
        f"- Equivalent race times if economy matched Daniels' model: {pr(last.fitness_vo2)}",
        f"- Performance anchor (best HR-confirmed hard effort, 90 d): "
        + (f"**VDOT {pv:.1f}** → {pr(pv)}" if pd.notna(pv) else "none (no efforts ≥ ~88% HRmax)"),
        f"- Calibration: hard-effort VDOT / HR index on the same dates = {R.attrs['calib']:.3f} (n={R.attrs['calib_n']}) "
        f"→ performance-calibrated index **{last.fitness_vo2 * R.attrs['calib']:.1f}** → {pr(last.fitness_vo2 * R.attrs['calib'])}",
        "", "### Sensitivity to HR anchors (weighted mean of last 42 d runs, heat-adjusted)", "", "| RHR | HRmax | index |", "|---|---|---|",
        *[f"| {r_} | {m_} | {v:.1f} |" for (r_, m_), v in sens.items()], "",
        "## Monthly (eligible runs)", "", m.round(2).to_markdown(), "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "analyze"])
    ap.add_argument("--user", default=os.environ.get("COROS_USER_ID"), help="COROS user id in FIT URLs (or $COROS_USER_ID)")
    ap.add_argument("--hrmax", type=float, default=197)
    ap.add_argument("--rhr", type=float, default=45)
    a = ap.parse_args()
    {"fetch": fetch, "analyze": analyze}[a.cmd](a)
