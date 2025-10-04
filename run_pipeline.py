import os, numpy as np, pandas as pd, jax.numpy as jnp
from lightweight_mmm import lightweight_mmm, preprocessing

np.random.seed(42)
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# -----------------------------
# Fourier helpers & utilities
# -----------------------------
def fourier_series(t, period, order):
    X = []
    for k in range(1, order+1):
        X.append(np.sin(2*np.pi*k*t/period))
        X.append(np.cos(2*np.pi*k*t/period))
    return np.column_stack(X).astype("float32")

def credible_interval(samples, level=0.8):
    lo = (1 - level) / 2
    hi = 1 - lo
    q = np.quantile(samples, [lo, 0.5, hi], axis=0)
    return float(q[0]), float(q[1]), float(q[2])

def normalize_alloc(vec):
    s = np.sum(vec)
    if s <= 0: raise ValueError("Allocations must sum to > 0")
    return vec / s

def build_allocated_media(media_unscaled, shares, budget_mult=1.0):
    shares = normalize_alloc(np.array(shares, dtype="float32"))
    totals = media_unscaled.sum(axis=1, keepdims=True)             # (T,1)
    return (budget_mult * totals * shares[None, :]).astype("float32")

# -----------------------------
# 1) Load YOUR Robyn CSV
# -----------------------------
df = pd.read_csv("Robyn_simulated_weekly.csv", parse_dates=["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)
df["Week"] = np.arange(1, len(df)+1)

# Target
y = df["revenue"].astype("float32").to_numpy()

# Media spends (use *_S as spends)
media_cols = ["tv_S","ooh_S","print_S","facebook_S","search_S"]
# Include newsletter if you want it treated like a spend channel:
if "newsletter" in df.columns:
    media_cols.append("newsletter")

media_raw = df[media_cols].fillna(0).astype("float32").to_numpy()
# media_raw = df[media_cols].fillna(0).astype("float32").to_numpy()
total_costs = media_raw.sum(axis=0).astype("float32")
# Controls (non-spend signals)
extra_cols = [c for c in ["facebook_I","search_clicks_P","competitor_sales_B"] if c in df.columns]
extra_ctrl = df[extra_cols].fillna(0).astype("float32").to_numpy() if extra_cols else None

# Optional: one-hot event types if any (many rows are "na"; we skip if all NA)
if "events" in df.columns:
    ev = df["events"].fillna("na").astype(str)
    if ev.nunique() > 1 and not (ev.nunique()==1 and ev.unique()[0].lower()=="na"):
        event_dum = pd.get_dummies(ev, prefix="event", drop_first=True).astype("float32").to_numpy()
    else:
        event_dum = None
else:
    event_dum = None

# Seasonality (Strong-style Fourier: annual, order=3)
t = np.arange(len(df), dtype=int)
fourier = fourier_series(t, period=52, order=3)

# Combine extras
extras = [fourier]
if extra_ctrl is not None: extras.append(extra_ctrl)
if event_dum is not None: extras.append(event_dum)
extra_mat = np.hstack(extras).astype("float32")

# -----------------------------
# 2) Scale & fit LightweightMMM
# -----------------------------
media_scaler  = preprocessing.CustomScaler(divide_operation=jnp.mean, multiply_by=1.0)
target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean, multiply_by=1.0)
extra_scaler  = preprocessing.CustomScaler(divide_operation=jnp.std,  multiply_by=1.0)

media  = media_scaler.fit_transform(media_raw)
y_sc   = target_scaler.fit_transform(y)
extra  = extra_scaler.fit_transform(extra_mat)

import numpyro
numpyro.set_host_device_count(1)  # optional; silences parallel-chains warning on CPU

mmm = lightweight_mmm.LightweightMMM(model_name="hill_adstock")
mmm.fit(
    media=media,              # scaled spends
    target=y_sc,              # scaled revenue
    total_costs=total_costs,  # UNscaled weekly total spend
    extra_features=extra,     # scaled Fourier + controls
    number_warmup=600,
    number_samples=1000,
)
# -----------------------------
# 3) Baseline predictions & exports
# -----------------------------
pred_sc = mmm.predict(media=media, extra_features=extra)               # (draws, T)
pred    = pred_sc * target_scaler.divide_operation(y)                  # back to original scale
pred_med = np.median(pred, axis=0).astype("float32")

pred_df = pd.DataFrame({
    "Week": df["Week"].to_numpy(),
    "DATE": df["DATE"].dt.date.astype(str),
    "Actual": y,
    "Predicted_Median": pred_med
})
pred_df.to_csv(os.path.join(EXPORT_DIR, "predictions.csv"), index=False)

baseline_total = float(pred_med.sum())

# Posterior metrics (if available in your LwMMM version)
try:
    pd.DataFrame(mmm.get_posterior_metrics(metric="roi")).to_csv(
        os.path.join(EXPORT_DIR,"channel_roi.csv"), index=False)
except Exception:
    pass
try:
    pd.DataFrame(mmm.get_posterior_metrics(metric="response_curves")).to_csv(
        os.path.join(EXPORT_DIR,"response_curves.csv"), index=False)
except Exception:
    pass

# -----------------------------
# 4) Scenario grid for Tableau sliders
# -----------------------------
steps = list(range(0, 101, 10))
budget_mults = [0.8, 1.0, 1.2]
scenarios = []

def sums_to_100(a,b,c,d,e,f=0, tol=1e-6):
    return abs((a+b+c+d+e+f) - 100) <= tol

for tv in steps:
    for ooh in steps:
        for prt in steps:
            for fb in steps:
                for srch in steps:
                    rest = 100 - (tv+ooh+prt+fb+srch)
                    if "newsletter" in media_cols:
                        for news in steps:
                            if not sums_to_100(tv,ooh,prt,fb,srch,news): continue
                            shares = np.array([tv,ooh,prt,fb,srch,news], dtype="float32")/100.0
                            for bm in budget_mults:
                                media_new_un = build_allocated_media(media_raw, shares, budget_mult=bm)
                                media_new = media_scaler.transform(media_new_un)
                                pred_new_sc = mmm.predict(media=media_new, extra_features=extra)
                                pred_new = pred_new_sc * target_scaler.divide_operation(y)
                                lo, med, hi = credible_interval(pred_new.sum(axis=1))
                                scenarios.append({
                                    "tv":tv,"ooh":ooh,"print":prt,"facebook":fb,"search":srch,"newsletter":int(shares.size==6)*int(shares[5]*100),
                                    "BudgetMult":bm,
                                    "Pred_Total_Lo":lo,"Pred_Total_Med":med,"Pred_Total_Hi":hi,
                                    "Baseline_Total":baseline_total,
                                    "Lift_Abs": med - baseline_total,
                                    "Lift_Pct": 100*(med - baseline_total)/baseline_total,
                                    "ScenarioKey": f"{tv}-{ooh}-{prt}-{fb}-{srch}-{int(shares.size==6 and shares[5]*100)}-{bm}"
                                })
                    else:
                        if not sums_to_100(tv,ooh,prt,fb,srch): continue
                        shares = np.array([tv,ooh,prt,fb,srch], dtype="float32")/100.0
                        for bm in budget_mults:
                            media_new_un = build_allocated_media(media_raw, shares, budget_mult=bm)
                            media_new = media_scaler.transform(media_new_un)
                            pred_new_sc = mmm.predict(media=media_new, extra_features=extra)
                            pred_new = pred_new_sc * target_scaler.divide_operation(y)
                            lo, med, hi = credible_interval(pred_new.sum(axis=1))
                            scenarios.append({
                                "tv":tv,"ooh":ooh,"print":prt,"facebook":fb,"search":srch,
                                "BudgetMult":bm,
                                "Pred_Total_Lo":lo,"Pred_Total_Med":med,"Pred_Total_HI":hi if 'hi' in locals() else hi,
                                "Baseline_Total":baseline_total,
                                "Lift_Abs": med - baseline_total,
                                "Lift_Pct": 100*(med - baseline_total)/baseline_total,
                                "ScenarioKey": f"{tv}-{ooh}-{prt}-{fb}-{srch}-{bm}"
                            })

pd.DataFrame(scenarios).to_csv(os.path.join(EXPORT_DIR, "scenarios.csv"), index=False)

# (Optional) weekly series for a few “hero” mixes
hero_mixes = [
    {"tv":30,"ooh":10,"print":10,"facebook":30,"search":20},
    {"tv":20,"ooh":10,"print":10,"facebook":40,"search":20},
    {"tv":10,"ooh":10,"print":10,"facebook":45,"search":25},
]
weekly_rows = []
for w in hero_mixes:
    weights = np.array([w["tv"],w["ooh"],w["print"],w["facebook"],w["search"]], dtype="float32")
    if "newsletter" in media_cols:
        weights = np.append(weights, 0.0)
    shares = weights / weights.sum()
    media_new_un = build_allocated_media(media_raw, shares, 1.0)
    media_new = media_scaler.transform(media_new_un)
    pred_new_sc = mmm.predict(media=media_new, extra_features=extra)
    pred_new = pred_new_sc * target_scaler.divide_operation(y)
    weekly_med = np.median(pred_new, axis=0)
    for wk, base, scen in zip(df["Week"], pred_med, weekly_med):
        weekly_rows.append({"Mix": str(w), "Week": int(wk),
                            "Baseline_Weekly": float(base),
                            "Scenario_Weekly": float(scen)})

pd.DataFrame(weekly_rows).to_csv(os.path.join(EXPORT_DIR, "scenario_weekly.csv"), index=False)

print("✅ Exports written to ./exports : predictions.csv, scenarios.csv, (roi/curves if available), scenario_weekly.csv")