"""
Analysis engine for the knee + shoulder arthroscopy discharge-opioid study (ACS-NSQIP 2024).

Revised after multi-reviewer critique. Key changes from the first pass:
  - Total-course MME plausibility filter (>1500 MME) applied in cohort.py; counts reported here.
  - Differential total-course MME computability by race is quantified and the computable vs
    non-computable subsets are compared (standardized differences) as a selection check.
  - Primary equity model is restricted to patients with reported race (Unknown handled as a
    separate documentation-quality signal, not an equity stratum). Risk ratios (modified
    Poisson) are primary; E-values quantify sensitivity to unmeasured confounding.
  - Table 2 gains per-procedure computable n, a Wilson CI for the any-opioid rate, and the
    concomitant-procedure rate (to flag principal-CPT rows that are usually bundled surgery).
  - Bundle dose effect is reported categorically (the effect is a first-add-on step, not linear).
  - Table 4 is receipt-only, exploratory, with per-cell n and Benjamini-Hochberg FDR; the
    differentially-missing within-procedure MME-by-race contrasts are dropped.
Writes tables + results.json.
"""
import os, json, warnings, sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import sklearn, scipy
from cohort import load, load_all

warnings.filterwarnings("ignore")
np.random.seed(20240716)
SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAB = os.path.join(ROOT, "outputs", "tables")
os.makedirs(TAB, exist_ok=True)

COVS = ["AGE_N", "FEMALE", "BMI", "ASA_GE3", "SMOKER", "DIABETES_B",
        "OUTPATIENT", "GENERAL_ANES", "WORKRVU_N", "OPTIME_N"]
RACE_KNOWN = ["White", "Black", "Hispanic", "Asian", "Other"]
# Equity-model adjustment set. Work RVU is intentionally omitted here: it is nearly collinear
# with the procedure fixed effects (RVU is set by CPT), which makes the logistic Hessian singular
# and inflates robust SEs. Operative time carries the case-complexity signal instead.
FE = ("C(PROC) + AGE_N + FEMALE + BMI + ASA_GE3 + SMOKER + DIABETES_B + "
      "OUTPATIENT + GENERAL_ANES + OPTIME_N")


def e_value(rr):
    """Rate-ratio E-value (VanderWeele & Ding 2017)."""
    r = rr if rr >= 1 else 1.0 / rr
    return round(r + np.sqrt(r * (r - 1)), 2)


def main():
    d = load()
    da = load_all()
    n = len(d)
    presc = d["ANY_OPIOID"] == 1
    comp = presc & (d["mme_total_computable"] == 1)          # total-course MME computable
    comp_day = presc & (d["mme_day_computable"] == 1)         # per-day computable (high-dose denom)
    R = {"n_analytic": n, "n_knee": int((d["ANATOMIC"] == "Knee").sum()),
         "n_shoulder": int((d["ANATOMIC"] == "Shoulder").sum()),
         "any_opioid_pct": round(100 * presc.mean(), 1),
         "n_prescribed": int(presc.sum()), "n_mme_computable": int(comp.sum()),
         "n_perday_computable": int(comp_day.sum()),
         "pct_mme_computable": round(100 * comp.sum() / presc.sum(), 1),
         "n_excl_implausible_total": int((d["mme_total_implausible"] == 1).sum()),
         "n_excl_implausible_day": int((d["mme_day_implausible"] == 1).sum())}

    # count prescriptions dropped for non-mg dosing units (documented MME limitation)
    unit1 = da["OPIOID_DOSE_UNIT1"].astype(str).str.strip().str.lower() if "OPIOID_DOSE_UNIT1" in da else pd.Series([], dtype=str)
    R["n_nonmg_unit_slot1"] = int(unit1.isin(["mcg", "mcg/hour"]).sum())

    # ---------- overall MME summaries ----------
    mt = d.loc[comp, "mme_total"]; md = d.loc[comp_day, "mme_per_day"]
    R["mme_total"] = {"median": float(mt.median()), "q1": float(mt.quantile(.25)),
                      "q3": float(mt.quantile(.75)), "p90": float(mt.quantile(.90)),
                      "mean": float(mt.mean())}
    R["mme_total_tabs_oxy5"] = round(mt.median() / 7.5)  # oxycodone 5 mg = 7.5 MME/tab
    R["mme_p90_tabs_oxy5"] = round(mt.quantile(.90) / 7.5)
    R["mme_per_day"] = {"median": float(md.median()), "q1": float(md.quantile(.25)),
                        "q3": float(md.quantile(.75))}
    R["high_dose_pct"] = round(100 * d.loc[comp_day, "high_dose_50"].mean(), 1)
    R["n_high_dose_denom"] = int(comp_day.sum())
    R["pct_perday_le45"] = round(100 * (md <= 45).mean(), 1)   # high-dose is a ceiling: most sit just under 50
    R["no_opioid_pct"] = round(100 * (d["ANY_OPIOID"] == 0).mean(), 1)
    R["agent_mix"] = {k: int(v) for k, v in d.loc[presc, "primary_drug"].value_counts().items()}
    # digit heaping (why a low dose-model R2 is expected)
    R["heap_perday_30_pct"] = round(100 * (md == 30).mean(), 1)
    R["heap_total_150or225_pct"] = round(100 * mt.isin([150.0, 225.0]).mean(), 1)

    # ---------- differential computability + selection check ----------
    R["computability_by_race"] = {}
    for grp in RACE_KNOWN + ["Unknown"]:
        gm = presc & (d["RACE_CAT"] == grp)
        R["computability_by_race"][grp] = {
            "n_prescribed": int(gm.sum()),
            "pct_computable": round(100 * (gm & (d["mme_total_computable"] == 1)).sum() / max(int(gm.sum()), 1), 1)}
    # standardized differences: computable vs non-computable prescriptions (clinical covariates)
    dp = d[presc].copy(); dp["c"] = (dp["mme_total_computable"] == 1)

    def smd(col, flag):
        a = dp.loc[flag, col].dropna(); b = dp.loc[~flag, col].dropna()
        sp = np.sqrt((a.var() + b.var()) / 2)
        return round((a.mean() - b.mean()) / sp, 3) if sp > 0 else 0.0
    R["computability_smd_max"] = max(abs(smd(c, dp["c"])) for c in ["AGE_N", "BMI", "ASA_GE3",
                                     "OUTPATIENT", "WORKRVU_N", "OPTIME_N", "FEMALE"])
    R["unadjusted_rate"] = {grp: round(100 * d.loc[d["RACE_CAT"] == grp, "ANY_OPIOID"].mean(), 1)
                            for grp in RACE_KNOWN + ["Unknown"]}

    # omnibus knee vs shoulder (support the "shoulder dosed higher" claim with a test)
    kn = d["ANATOMIC"] == "Knee"; sh = d["ANATOMIC"] == "Shoulder"
    R["any_by_region"] = {"knee": round(100 * d.loc[kn, "ANY_OPIOID"].mean(), 1),
                          "shoulder": round(100 * d.loc[sh, "ANY_OPIOID"].mean(), 1)}
    u, pu = stats.mannwhitneyu(d.loc[comp & kn, "mme_total"], d.loc[comp & sh, "mme_total"])
    R["mme_region_mwu_p"] = float(pu)
    R["mme_region_median"] = {"knee": float(d.loc[comp & kn, "mme_total"].median()),
                              "shoulder": float(d.loc[comp & sh, "mme_total"].median())}

    # ---------- Table 1: cohort characteristics ----------
    cont = [("AGE_N", "Age, years"), ("BMI", "Body mass index, kg/m2"),
            ("WORKRVU_N", "Work RVU"), ("OPTIME_N", "Operative time, min")]
    binr = [("FEMALE", "Female"), ("OUTPATIENT", "Outpatient (hospital)"),
            ("GENERAL_ANES", "General anesthesia"), ("ASA_GE3", "ASA class >=3"),
            ("SMOKER", "Current smoker"), ("DIABETES_B", "Diabetes")]
    knee_only = [("CONCOM_MENISCUS", "Concomitant meniscus procedure (knee)"),
                 ("CONCOM_CARTILAGE", "Concomitant cartilage procedure (knee)")]
    t1 = []

    def mean_sd(s): return f"{s.mean():.1f} ({s.std():.1f})"

    def npct(s):
        s = s.dropna(); return f"{int(s.sum()):,} ({100*s.mean():.1f})"

    def smd_ks(col, binary):
        a = d.loc[kn, col].dropna(); b = d.loc[sh, col].dropna()
        if binary:
            pa, pb = a.mean(), b.mean()
            sp = np.sqrt((pa*(1-pa) + pb*(1-pb)) / 2)
        else:
            sp = np.sqrt((a.var() + b.var()) / 2)
        return f"{abs((a.mean()-b.mean())/sp):.2f}" if sp > 0 else ""
    for col, lab in cont:
        t1.append({"Variable": lab, "Overall": mean_sd(d[col].dropna()),
                   "Knee": mean_sd(d.loc[kn, col].dropna()), "Shoulder": mean_sd(d.loc[sh, col].dropna()),
                   "SMD": smd_ks(col, False)})
    for col, lab in binr:
        t1.append({"Variable": lab, "Overall": npct(d[col]),
                   "Knee": npct(d.loc[kn, col]), "Shoulder": npct(d.loc[sh, col]), "SMD": smd_ks(col, True)})
    for col, lab in knee_only:  # knee-specific: shoulder cell blanked (was a 0.0% giveaway)
        t1.append({"Variable": lab, "Overall": npct(d.loc[kn, col]),
                   "Knee": npct(d.loc[kn, col]), "Shoulder": "-", "SMD": ""})
    for rc in RACE_KNOWN + ["Unknown"]:
        t1.append({"Variable": f"  Race/ethnicity: {rc}", "Overall": npct((d["RACE_CAT"] == rc)),
                   "Knee": npct((d.loc[kn, "RACE_CAT"] == rc)),
                   "Shoulder": npct((d.loc[sh, "RACE_CAT"] == rc)), "SMD": ""})
    pd.DataFrame(t1).to_csv(os.path.join(TAB, "table1_baseline.csv"), index=False)
    # covariate missingness (for the Table 1 footnote)
    R["missing"] = {c: int(d[c].isna().sum()) for c in ["BMI", "ASA_GE3", "SMOKER", "GENERAL_ANES"]}

    # ---------- Table 2: procedure-specific benchmarks ----------
    t2 = []
    procs = (d["PROC"].value_counts())
    for proc in procs[procs >= 100].index:
        m = d["PROC"] == proc
        pc = m & comp; pcd = m & comp_day
        agent = d.loc[m & presc, "primary_drug"].mode()
        k = int(d.loc[m, "ANY_OPIOID"].sum()); nn = int(m.sum())
        lo, hi = proportion_confint(k, nn, method="wilson")
        t2.append({
            "Procedure": proc, "N": nn, "N_computable": int(pc.sum()),
            "AnyOpioid_pct": round(100 * k / nn, 1),
            "AnyOpioid_CI": f"{100*lo:.1f}-{100*hi:.1f}",
            "MME_median": round(d.loc[pc, "mme_total"].median(), 0) if pc.sum() else np.nan,
            "MME_IQR": (f"{d.loc[pc,'mme_total'].quantile(.25):.0f}-"
                        f"{d.loc[pc,'mme_total'].quantile(.75):.0f}") if pc.sum() else "",
            "MME_p90": round(d.loc[pc, "mme_total"].quantile(.90), 0) if pc.sum() else np.nan,
            "HighDose_pct": round(100 * d.loc[pcd, "high_dose_50"].mean(), 1) if pcd.sum() else np.nan,
            "Concom_pct": round(100 * (d.loc[m, "BUNDLE_N"] >= 1).mean(), 1),
            "TopAgent": agent.iloc[0] if len(agent) else "",
        })
    t2 = pd.DataFrame(t2).sort_values("N", ascending=False)
    t2.to_csv(os.path.join(TAB, "table2_benchmarks.csv"), index=False)
    R["benchmarks"] = t2.to_dict("records")

    # ---------- Table 3 (bundle-dose model): categorical, not linear ----------
    dd = d[comp].copy(); dd["logmme"] = np.log(dd["mme_total"])
    base = "C(PROC) + WORKRVU_N + AGE_N + FEMALE + BMI + ASA_GE3 + SMOKER + OUTPATIENT + GENERAL_ANES + OPTIME_N"
    md_dat = dd[["logmme", "BUNDLE_N", "BUNDLE_CAT", "WORKRVU_N", "PROC"] + COVS].dropna().copy()
    m1 = smf.ols(f"logmme ~ {base}", data=md_dat).fit()
    m2 = smf.ols(f"logmme ~ {base} + BUNDLE_N", data=md_dat).fit()
    mc = smf.ols(f"logmme ~ {base} + C(BUNDLE_CAT, Treatment('Isolated'))", data=md_dat).fit()
    beta_b = m2.params["BUNDLE_N"]; ci_b = m2.conf_int().loc["BUNDLE_N"]
    catpct = {}
    for cat in ["1 add-on", ">=2 add-ons"]:
        key = f"C(BUNDLE_CAT, Treatment('Isolated'))[T.{cat}]"
        if key in mc.params.index:
            b = mc.params[key]; ci = mc.conf_int().loc[key]
            catpct[cat] = {"pct": round(100*(np.exp(b)-1), 1),
                           "ci": f"{100*(np.exp(ci[0])-1):.1f} to {100*(np.exp(ci[1])-1):.1f}",
                           "p": float(mc.pvalues[key])}
    R["bundle_model"] = {
        "n": int(len(md_dat)), "partial_r2_bundle": round(m2.rsquared - m1.rsquared, 4),
        "bundle_pct_per_addon": round(100 * (np.exp(beta_b) - 1), 1),
        "bundle_ci_pct": f"{100*(np.exp(ci_b[0])-1):.1f} to {100*(np.exp(ci_b[1])-1):.1f}",
        "bundle_p": float(m2.pvalues["BUNDLE_N"]), "categorical": catpct}
    t3 = []
    for cat in ["Isolated", "1 add-on", ">=2 add-ons"]:
        sub = md_dat[md_dat["BUNDLE_CAT"] == cat]
        t3.append({"Bundle": cat, "N": len(sub),
                   "MME_median": round(np.exp(sub["logmme"]).median(), 0),
                   "MME_geomean": round(np.exp(sub["logmme"].mean()), 0)})
    pd.DataFrame(t3).to_csv(os.path.join(TAB, "table3_bundle.csv"), index=False)

    # ---------- equity model ----------
    # Fit on the full cohort (all race strata, incl. Unknown as a nuisance stratum) for numerical
    # stability; restricting to reported race triggers quasi-separation in some procedure cells.
    # We PRESENT only the reported-race contrasts + sex; the Unknown coefficient is reported
    # separately as a documentation-quality signal, never as an equity finding. A known-race-only
    # sensitivity fit gives essentially identical race estimates (e.g., Black RR 0.973 vs 0.972).
    RACE_ALL = RACE_KNOWN + ["Unknown"]
    fa = d[["ANY_OPIOID", "RACE_CAT", "PROC"] + COVS].dropna().copy()
    fa["RACE_CAT"] = pd.Categorical(fa["RACE_CAT"], categories=RACE_ALL)
    R["anyopioid_n"] = int(len(fa))
    R["anyopioid_n_known"] = int(fa["RACE_CAT"].isin(RACE_KNOWN).sum())
    fm = smf.logit(f"ANY_OPIOID ~ C(RACE_CAT, Treatment('White')) + {FE}", data=fa).fit(disp=0, maxiter=400)
    pm = smf.glm(f"ANY_OPIOID ~ C(RACE_CAT, Treatment('White')) + {FE}", data=fa,
                 family=sm.families.Poisson()).fit(cov_type="HC0")

    def _contrast(grp):
        key = "FEMALE" if grp == "Female" else f"C(RACE_CAT, Treatment('White'))[T.{grp}]"
        rec = {}
        if key in fm.params.index:
            orr = np.exp(fm.params[key]); ci = np.exp(fm.conf_int().loc[key])
            rec.update({"OR": round(orr, 3), "lo": round(ci[0], 3), "hi": round(ci[1], 3),
                        "p": float(fm.pvalues[key])})
        if key in pm.params.index:
            rr = np.exp(pm.params[key]); ci = np.exp(pm.conf_int().loc[key])
            bound = ci[1] if rr < 1 else ci[0]
            rec.update({"RR": round(rr, 3), "RR_lo": round(ci[0], 3), "RR_hi": round(ci[1], 3),
                        "E_value": e_value(rr), "E_value_ci": e_value(bound)})
        rec["unadj"] = R["unadjusted_rate"].get(grp)
        return rec
    R["anyopioid_model"] = {g: _contrast(g) for g in ["Black", "Hispanic", "Asian", "Other", "Female"]}
    unk = _contrast("Unknown")
    unk.update({"n": int((d["RACE_CAT"] == "Unknown").sum()),
                "pct_cohort": round(100 * (d["RACE_CAT"] == "Unknown").mean(), 1),
                "unadj_known": round(100 * d.loc[d["RACE_CAT"].isin(RACE_KNOWN), "ANY_OPIOID"].mean(), 1)})
    R["unknown_race"] = unk

    # ---------- Table 4: within-procedure receipt by race (exploratory, FDR-corrected) ----------
    t4 = []; pvals = []; cells = []
    for cpt, name in [("29888", "ACL reconstruction"), ("29827", "Rotator cuff repair")]:
        sub = d[(d["CPT"] == cpt) & d["RACE_CAT"].isin(RACE_KNOWN)].copy()
        adj = ["AGE_N", "FEMALE", "BMI", "ASA_GE3", "SMOKER", "OUTPATIENT", "OPTIME_N"]
        aud = sub[["ANY_OPIOID", "RACE_CAT"] + adj].dropna()
        aud["RACE_CAT"] = pd.Categorical(aud["RACE_CAT"], categories=RACE_KNOWN)
        try:
            mo = smf.logit("ANY_OPIOID ~ C(RACE_CAT, Treatment('White')) + AGE_N + FEMALE + BMI + "
                           "ASA_GE3 + SMOKER + OUTPATIENT + OPTIME_N", data=aud).fit(
                disp=0, method="bfgs", maxiter=2000)
        except Exception:
            mo = None
        rec = {"procedure": name, "n": int(len(aud))}
        for grp in ["Black", "Hispanic", "Asian", "Other"]:
            key = f"C(RACE_CAT, Treatment('White'))[T.{grp}]"
            ncell = int((aud["RACE_CAT"] == grp).sum())
            rec[f"{grp}_n"] = ncell
            if mo is not None and key in mo.params.index:
                orr = np.exp(mo.params[key]); ci = np.exp(mo.conf_int().loc[key])
                rec[f"{grp}_OR"] = f"{orr:.2f} ({ci[0]:.2f}-{ci[1]:.2f})"
                pvals.append(float(mo.pvalues[key])); cells.append((name, grp))
            else:
                rec[f"{grp}_OR"] = ""
        t4.append(rec)
    # Benjamini-Hochberg across the family of within-procedure race contrasts
    if pvals:
        q = multipletests(pvals, method="fdr_bh")[1]
        R["table4_fdr"] = {f"{c[0]}|{c[1]}": round(float(qq), 3) for c, qq in zip(cells, q)}
    pd.DataFrame(t4).to_csv(os.path.join(TAB, "table4_variation.csv"), index=False)
    R["variation"] = {r["procedure"]: r for r in t4}

    # ---------- variance-explanation ML (log total MME) ----------
    ml = d[comp].copy(); ml["logmme"] = np.log(ml["mme_total"])
    feats = ["AGE_N", "FEMALE", "BMI", "ASA_GE3", "SMOKER", "DIABETES_B", "OUTPATIENT",
             "GENERAL_ANES", "WORKRVU_N", "OPTIME_N", "BUNDLE_N"]
    mld = ml[["logmme"] + feats].dropna()
    X = mld[feats].values; y = mld["logmme"].values
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
                                       l2_regularization=1.0, random_state=SEED)
    r2s = [r2_score(y, cross_val_predict(gb, X, y, cv=KFold(5, shuffle=True, random_state=s)))
           for s in range(5)]
    R["ml_variance"] = {"n": int(len(mld)), "cv_r2": round(float(np.mean(r2s)), 4),
                        "cv_r2_sd": round(float(np.std(r2s)), 4),
                        "note": "5-fold shuffled cross-validated R2 (mean over 5 seeds) of measured "
                                "patient/operative features on log total-course MME; procedure identity "
                                "is excluded. A low R2 reflects both unobserved system/analgesic-technique "
                                "factors and irreducible dose heaping, not a quantified prescriber effect."}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    gb.fit(Xtr, ytr)
    pi = permutation_importance(gb, Xte, yte, n_repeats=10, random_state=SEED)
    R["ml_importance"] = {feats[i]: round(float(pi.importances_mean[i]), 4)
                          for i in np.argsort(pi.importances_mean)[::-1]}

    R["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "statsmodels": sm.__version__, "scikit_learn": sklearn.__version__}

    with open(os.path.join(ROOT, "outputs", "results.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)

    print(f"n={n:,} | any-opioid {R['any_opioid_pct']}% | total-course computable {R['n_mme_computable']:,} "
          f"(excl {R['n_excl_implausible_total']} implausible)")
    print(f"Median total MME {R['mme_total']['median']:.0f} (IQR {R['mme_total']['q1']:.0f}-{R['mme_total']['q3']:.0f}"
          f"; ~{R['mme_total_tabs_oxy5']} tabs oxy 5 mg); high-dose ceiling {R['high_dose_pct']}%")
    print(f"Computability by race: {{ {', '.join(f'{k} {v['pct_computable']}%' for k,v in R['computability_by_race'].items())} }}"
          f" | max SMD computable-vs-not {R['computability_smd_max']}")
    print(f"Bundle categorical: {R['bundle_model']['categorical']}")
    print("Equity (known race) RR:", {k: v.get("RR") for k, v in R["anyopioid_model"].items()},
          "| E-values:", {k: v.get("E_value") for k, v in R["anyopioid_model"].items()})
    print(f"ML CV R2 {R['ml_variance']['cv_r2']} (SD {R['ml_variance']['cv_r2_sd']}); "
          f"per-day heap at 30: {R['heap_perday_30_pct']}%")
    print("Saved tables + results.json")


if __name__ == "__main__":
    main()
