"""
Morphine-milligram-equivalent (MME) computation for the ACS-NSQIP 2024 discharge-opioid module.

Standard CDC/CMS oral MME conversion factors (mg opioid -> mg oral morphine equivalent).
The module records, per drug slot 1..5: OPIOID_TYPE, OPIOID_METHOD, OPIOID_DOSE (per-unit mg),
OPIOID_DOSE_UNIT, OPIOID_DOSE_FREQ (dosing interval), OPIOID_DOSE_PSC (quantity dispensed).

Two MME summaries are computed per case:
  - mme_per_day  = sum over slots of  dose_mg * doses_per_day(freq) * factor      (prescribed daily intensity)
  - mme_total    = sum over slots of  dose_mg * quantity(PSC)       * factor      (total dispensed load)
High-dose prescribing (a CDC-defined concern threshold) is mme_per_day >= 50.
"""
import numpy as np
import pandas as pd

# CDC 2022 / CMS standard oral MME conversion factors, keyed by the leading drug word in OPIOID_TYPE.
MME_FACTOR = {
    "oxycodone": 1.5,
    "hydrocodone": 1.0,
    "hydromorphone": 4.0,
    "codeine": 0.15,
    "tramadol": 0.1,
    "oral morphine": 1.0,
    "morphine": 1.0,
    "oxymorphone": 3.0,
    "tapentadol": 0.4,
}

# dosing interval string -> doses per day
FREQ_PER_DAY = {
    "every hour": 24.0,
    "every 2 hours": 12.0,
    "every 2-4 hours": 8.0,
    "every 4 hours": 6.0,
    "every 4-6 hours": 4.8,
    "every 6 hours": 4.0,
    "every 6-8 hours": 3.4,
    "every 8 hours": 3.0,
    "every 12 hours (twice daily)": 2.0,
    "every 24 hours (daily)": 1.0,
}


def _factor(type_str):
    t = str(type_str).strip().lower()
    for key, fac in MME_FACTOR.items():
        if t.startswith(key):
            return fac
    return np.nan  # "Other" / unmapped -> not counted (documented limitation)


def _freq_per_day(freq_str):
    f = str(freq_str).strip().lower()
    return FREQ_PER_DAY.get(f, np.nan)


def _num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.strip().replace(
        {"": np.nan, "nan": np.nan, "-99": np.nan, "NULL": np.nan, "None": np.nan}), errors="coerce")


def compute_mme(df, nslots=5):
    """Return a frame with per-case mme_per_day, mme_total, n_drug_slots, primary drug/type flags,
    and computability flags. df must hold OPIOID_TYPE{i}/DOSE{i}/DOSE_FREQ{i}/DOSE_PSC{i}/DOSE_UNIT{i}."""
    idx = df.index
    per_day = pd.Series(0.0, index=idx)
    total = pd.Series(0.0, index=idx)
    any_day = pd.Series(False, index=idx)     # at least one slot contributed to per_day
    any_total = pd.Series(False, index=idx)    # at least one slot contributed to total
    nslot = pd.Series(0, index=idx)
    for i in range(1, nslots + 1):
        typ = df.get(f"OPIOID_TYPE{i}")
        if typ is None:
            continue
        has = typ.astype(str).str.strip().replace({"NULL": "", "nan": ""}) != ""
        nslot = nslot + has.astype(int)
        fac = typ.map(_factor)
        dose = _num(df.get(f"OPIOID_DOSE{i}"))
        unit = df.get(f"OPIOID_DOSE_UNIT{i}").astype(str).str.strip().str.lower()
        # only count mg-dosed oral agents; skip mcg/patch (rare) to avoid unit-mismatch errors
        dose = dose.where(unit.eq("mg"))
        dpd = df.get(f"OPIOID_DOSE_FREQ{i}").map(_freq_per_day)
        qty = _num(df.get(f"OPIOID_DOSE_PSC{i}"))

        contrib_day = dose * dpd * fac
        contrib_tot = dose * qty * fac
        m_day = contrib_day.notna()
        m_tot = contrib_tot.notna()
        per_day = per_day + contrib_day.where(m_day, 0.0)
        total = total + contrib_tot.where(m_tot, 0.0)
        any_day = any_day | m_day
        any_total = any_total | m_tot

    out = pd.DataFrame(index=idx)
    out["n_drug_slots"] = nslot
    out["mme_per_day"] = per_day.where(any_day, np.nan)
    out["mme_total"] = total.where(any_total, np.nan)
    out["mme_day_computable"] = any_day.astype(int)
    out["mme_total_computable"] = any_total.astype(int)
    out["high_dose_50"] = (out["mme_per_day"] >= 50).astype(float)
    out.loc[out["mme_per_day"].isna(), "high_dose_50"] = np.nan
    # primary drug (slot 1) family label
    t1 = df.get("OPIOID_TYPE1").astype(str).str.strip()
    fam = t1.str.split(" (", regex=False).str[0].str.strip()
    out["primary_drug"] = fam.where(fam.str.lower() != "null", "None")
    return out


if __name__ == "__main__":
    # smoke test on the cached premier cohort
    import os
    P = os.environ.get("SPORTS_OPIOID_CACHE", "premier_2024_ortho.pkl")
    if os.path.exists(P):
        d = pd.read_pickle(P)
        got = d[d["OPIOID_DISCHARGE"].astype(str).str.strip() == "Yes"]
        m = compute_mme(got)
        print(f"prescribed n={len(got):,}")
        print("mme_per_day computable:", int(m["mme_day_computable"].sum()),
              "| median:", round(m["mme_per_day"].median(), 1))
        print("mme_total computable:", int(m["mme_total_computable"].sum()),
              "| median:", round(m["mme_total"].median(), 1))
        print("high-dose(>=50/day) among computable:",
              round(100 * m["high_dose_50"].mean(), 1), "%")
        print("primary drug mix:\n", m["primary_drug"].value_counts().head(8).to_string())
    else:
        print("cohort not cached yet")
