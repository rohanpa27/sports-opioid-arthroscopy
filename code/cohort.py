"""
Single source of truth for the sports knee + shoulder arthroscopy discharge-opioid cohort
(ACS-NSQIP 2024). load() returns the analysis-ready cohort with the anatomic pool, procedure
label, discharge-opioid outcomes (any opioid, total-course and daily MME, high-dose flag,
primary agent), concomitant-procedure bundle burden, and covariates. strobe() gives the
selection ledger.

Total-course MME uses OPIOID_DOSE_PSC (total doses prescribed) x per-dose strength x CDC factor,
summed across up to five agent slots; daily MME uses dosing frequency (secondary, PRN-sensitive).
See mme.py for the conversion engine.
"""
import os
import numpy as np
import pandas as pd
import mme as MME

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

KNEE = {"29888", "29880", "29881", "29882", "29883", "29866", "29867", "29877", "29879", "29876", "29875"}
SHOULDER = {"29827", "29806", "29807", "29826", "29828", "29823", "29822", "29824"}

PROC_LABEL = {
    "29888": "ACL reconstruction", "29880": "Meniscectomy (medial and lateral)",
    "29881": "Meniscectomy (medial or lateral)", "29882": "Meniscus repair (medial or lateral)",
    "29883": "Meniscus repair (medial and lateral)", "29866": "Osteochondral autograft",
    "29867": "Osteochondral allograft", "29877": "Chondroplasty",
    "29879": "Microfracture/abrasion", "29876": "Synovectomy (major)", "29875": "Synovectomy (limited)",
    "29827": "Rotator cuff repair", "29806": "Capsulorrhaphy (instability)", "29807": "SLAP repair",
    "29826": "Subacromial decompression", "29828": "Biceps tenodesis",
    "29823": "Debridement (extensive)", "29822": "Debridement (limited)", "29824": "Distal claviculectomy",
}
CPT_FIELDS = ["CPT"] + [f"OTHERCPT{i}" for i in range(1, 11)] + [f"CONCPT{i}" for i in range(1, 11)]
MENISCUS = {"29880", "29881", "29882", "29883"}
CARTILAGE = {"29866", "29867", "29877", "29879"}


def _num(s):
    return pd.to_numeric(
        s.astype(str).str.strip().replace(
            {"": np.nan, "nan": np.nan, "-99": np.nan, "NULL": np.nan, "None": np.nan}),
        errors="coerce")


def _yes(s):
    v = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index)
    out[v == "yes"] = 1; out[v == "no"] = 0
    return out


def comp_positive(s):
    v = s.astype(str).str.strip().str.lower()
    return ~v.isin(["no complication", "", "nan", "none", "null"])


def _bundle(d):
    """Count distinct additional procedure codes across OTHERCPT/CONCPT fields (concomitant burden)."""
    cnt = pd.Series(0, index=d.index)
    has_men = pd.Series(False, index=d.index)
    has_cart = pd.Series(False, index=d.index)
    for f in [c for c in CPT_FIELDS if c != "CPT"]:
        if f in d.columns:
            v = d[f].astype(str).str.strip().replace({"NULL": "", "nan": "", "-99": ""})
            nonempty = v != ""
            cnt = cnt + nonempty.astype(int)
            has_men = has_men | v.isin(MENISCUS)
            has_cart = has_cart | v.isin(CARTILAGE)
    return cnt, has_men, has_cart


def _derive(d):
    d = d.copy()
    d["CPT"] = d["CPT"].astype(str).str.strip()
    d["ANATOMIC"] = np.where(d["CPT"].isin(KNEE), "Knee",
                             np.where(d["CPT"].isin(SHOULDER), "Shoulder", "Other"))
    d["PROC"] = d["CPT"].map(PROC_LABEL).fillna("Other")

    # --- concomitant bundle ---
    bcnt, has_men, has_cart = _bundle(d)
    d["BUNDLE_N"] = bcnt
    d["BUNDLE_CAT"] = np.where(bcnt == 0, "Isolated", np.where(bcnt == 1, "1 add-on", ">=2 add-ons"))
    d["CONCOM_MENISCUS"] = has_men.astype(int)
    d["CONCOM_CARTILAGE"] = has_cart.astype(int)

    # --- demographics ---
    d["AGE_N"] = _num(d["AGE"].astype(str).str.replace("90+", "90", regex=False))
    d["FEMALE"] = (d["SEX"].astype(str).str.strip().str.lower() == "female").astype(int)
    h = _num(d["HEIGHT"]); w = _num(d["WEIGHT"])
    d["BMI"] = (w * 0.453592) / ((h * 0.0254) ** 2)
    d.loc[(d["BMI"] < 12) | (d["BMI"] > 70), "BMI"] = np.nan
    d["OUTPATIENT"] = (d["INOUT"].astype(str).str.strip().str.lower() == "outpatient").astype(int)

    # race/ethnicity category (Hispanic ethnicity takes precedence; Unknown kept as a stratum)
    r = d["RACE_NEW"].astype(str).str.strip()
    eth = d["ETHNICITY_HISPANIC"].astype(str).str.strip().str.lower()
    cat = pd.Series("Other", index=d.index)
    cat[r.eq("White")] = "White"
    cat[r.eq("Black or African American")] = "Black"
    cat[r.eq("Asian")] = "Asian"
    cat[r.isin(["Unknown/Not Reported", "", "nan", "NULL"])] = "Unknown"
    cat[eth.eq("yes")] = "Hispanic"
    d["RACE_CAT"] = cat

    # --- comorbidity / operative covariates ---
    asa = d["ASACLAS"].astype(str).str.strip()
    d["ASA_N"] = pd.to_numeric(asa.str[0], errors="coerce")
    d.loc[asa.str.lower().str.startswith("none"), "ASA_N"] = np.nan
    d["ASA_GE3"] = (d["ASA_N"] >= 3).astype(float)
    d.loc[d["ASA_N"].isna(), "ASA_GE3"] = np.nan
    sm = d["SMOKE"].astype(str).str.strip().str.lower()
    d["SMOKER"] = np.where(sm == "yes", 1.0, np.where(sm == "no", 0.0, np.nan))
    dm = d["DIABETES"].astype(str).str.strip().str.lower()
    d["DIABETES_B"] = np.where(dm.isin(["insulin", "non-insulin"]), 1.0,
                               np.where(dm == "no", 0.0, np.nan))
    an = d["ANESTHES"].astype(str).str.strip().str.lower()
    d["GENERAL_ANES"] = np.where(an == "general", 1.0,
                                 np.where(an.isin(["", "nan", "none", "unknown"]), np.nan, 0.0))
    d["WORKRVU_N"] = _num(d["WORKRVU"]); d["OPTIME_N"] = _num(d["OPTIME"])

    # --- discharge opioid outcomes ---
    od = d["OPIOID_DISCHARGE"].astype(str).str.strip().str.lower()
    d["ANY_OPIOID"] = np.where(od == "yes", 1.0, np.where(od == "no", 0.0, np.nan))
    m = MME.compute_mme(d)
    for c in ["mme_per_day", "mme_total", "high_dose_50", "n_drug_slots",
              "mme_day_computable", "mme_total_computable", "primary_drug"]:
        d[c] = m[c].values
    # Plausibility filter: a discharge course above 1500 MME (>=200 tablets of oxycodone 5 mg)
    # or a labeled daily dose above 150 MME is treated as a data-entry error and set
    # non-computable. Documented in Methods; flags let analysis.py count the exclusions.
    d["mme_total_implausible"] = (d["mme_total"] > 1500).astype(int)
    d["mme_day_implausible"] = (d["mme_per_day"] > 150).astype(int)
    d.loc[d["mme_total_implausible"] == 1, ["mme_total", "mme_total_computable"]] = [np.nan, 0]
    d.loc[d["mme_day_implausible"] == 1, ["mme_per_day", "mme_day_computable", "high_dose_50"]] = [np.nan, 0, np.nan]
    d["log_mme_total"] = np.log(d["mme_total"].where(d["mme_total"] > 0))

    # --- 30-day outcomes (context; rare in this healthy cohort) ---
    for src, name in [("SUPINFEC", "SSI_SUP"), ("WNDINFD", "SSI_DEEP"), ("ORGSPCSSI", "SSI_ORG"),
                      ("OUPNEUMO", "PNEUMONIA"), ("PULEMBOL", "PE"), ("OTHDVT", "DVT"),
                      ("OTHSYSEP", "SEPSIS"), ("CNSCVA", "STROKE"), ("URNINFEC", "UTI"),
                      ("OTHBLEED", "TRANSFUSION")]:
        d[name] = comp_positive(d[src]).astype(int)
    d["VTE"] = ((d["PE"] == 1) | (d["DVT"] == 1)).astype(int)
    d["READMISSION"] = ((_yes(d["READMISSION1"]).fillna(0).eq(1))
                        | (_yes(d["UNPLANNEDREADMISSION1"]).fillna(0).eq(1))).astype(int)
    d["REOPERATION"] = _yes(d["REOPERATION1"]).fillna(0).astype(int)
    return d


def load_all():
    d = pd.read_pickle(os.path.join(DATA, "cohort_raw.pkl"))
    return _derive(d)


def load():
    """Analytic cohort: sports knee/shoulder arthroscopy with a valid OPIOID_DISCHARGE record."""
    d = load_all()
    keep = d["ANATOMIC"].isin(["Knee", "Shoulder"]) & d["ANY_OPIOID"].notna()
    return d[keep].reset_index(drop=True)


def strobe():
    d = load_all()
    n_all = len(d)
    n_valid = int((d["ANATOMIC"].isin(["Knee", "Shoulder"])).sum())
    an = load()
    presc = an["ANY_OPIOID"] == 1
    return {
        "total_2024_sports": n_all,
        "excl_bad_opioid": int((d["ANATOMIC"].isin(["Knee", "Shoulder"]) & d["ANY_OPIOID"].isna()).sum()),
        "analytic": len(an),
        "knee": int((an["ANATOMIC"] == "Knee").sum()),
        "shoulder": int((an["ANATOMIC"] == "Shoulder").sum()),
        "prescribed": int(presc.sum()),
        "not_prescribed": int((an["ANY_OPIOID"] == 0).sum()),
        "mme_computable": int((an.loc[presc, "mme_total_computable"] == 1).sum()),
    }


if __name__ == "__main__":
    d = load()
    print(f"Analytic n = {len(d):,}")
    print("Anatomic:", d["ANATOMIC"].value_counts().to_dict())
    presc = d["ANY_OPIOID"] == 1
    print(f"Any discharge opioid: {int(presc.sum()):,} ({100*presc.mean():.1f}%)")
    print(f"MME-total computable among prescribed: {int((d.loc[presc,'mme_total_computable']==1).sum()):,}")
    print(f"Median total MME (computable): {d.loc[presc,'mme_total'].median():.0f}")
    print(f"Median daily MME (computable): {d.loc[presc,'mme_per_day'].median():.0f}")
    print(f"High-dose (>=50 MME/day) among computable: {100*d.loc[presc,'high_dose_50'].mean():.1f}%")
    print("\nPrimary agent mix:\n", d.loc[presc, "primary_drug"].value_counts().head(8).to_string())
    print("\nAny-opioid rate by procedure (top):")
    g = d.groupby("PROC")["ANY_OPIOID"].agg(["mean", "size"]).sort_values("size", ascending=False)
    print((g.assign(rate=lambda x: (100*x["mean"]).round(1))[["rate", "size"]]).head(12).to_string())
    print("\nRace mix:", d["RACE_CAT"].value_counts().to_dict())
    print("Bundle cat:", d["BUNDLE_CAT"].value_counts().to_dict())
    print("STROBE:", strobe())
