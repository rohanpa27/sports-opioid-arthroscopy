"""
Extract the 2024 sports knee + shoulder arthroscopy cohort with the discharge-opioid module
and the concomitant-procedure bundle fields (OTHERCPT/CONCPT), for the opioid-prescribing study.
Writes data/cohort_raw.pkl.
"""
import sys, os, time
sys.path.insert(0, "/Users/rohan/Documents/NSQIP Raw Data/ACS NSQIP 2010-2024/_shared")
import puf_source as ps
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
os.makedirs(DATA, exist_ok=True)

KNEE = {"29888", "29880", "29881", "29882", "29883", "29866", "29867", "29877", "29879", "29876", "29875"}
SHOULDER = {"29827", "29806", "29807", "29826", "29828", "29823", "29822", "29824"}
CPTS = KNEE | SHOULDER

DEMO = ["CASEID", "PUFYEAR", "OPERYR", "AGE", "SEX", "RACE_NEW", "ETHNICITY_HISPANIC",
        "HEIGHT", "WEIGHT", "INOUT", "TRANST"]
PROC = ["CPT", "PRNCPTX", "WORKRVU", "SURGSPEC", "ANESTHES", "OPTIME", "TOTHLOS"]
CONC = [f"OTHERCPT{i}" for i in range(1, 11)] + [f"CONCPT{i}" for i in range(1, 11)]
COM = ["ASACLAS", "FNSTATUS2", "DIABETES", "SMOKE", "HXCOPD", "HXCHF", "HYPERMED",
       "DISCANCR", "STEROID", "BLEEDDIS", "PRSEPIS"]
OUTC = ["SUPINFEC", "WNDINFD", "ORGSPCSSI", "OUPNEUMO", "PULEMBOL", "OTHDVT", "OTHSYSEP",
        "OTHBLEED", "CNSCVA", "CDARREST", "URNINFEC", "DOPERTOD", "DISCHDEST", "MORTPROB", "MORBPROB",
        "READMISSION1", "UNPLANNEDREADMISSION1", "REOPERATION1"]
OP = (["OPIOID_DISCHARGE"]
      + [f"OPIOID_TYPE{i}" for i in range(1, 6)]
      + [f"OPIOID_METHOD{i}" for i in range(1, 6)]
      + [f"OPIOID_DOSE{i}" for i in range(1, 6)]
      + [f"OPIOID_DOSE_UNIT{i}" for i in range(1, 6)]
      + [f"OPIOID_DOSE_FREQ{i}" for i in range(1, 6)]
      + [f"OPIOID_DOSE_PSC{i}" for i in range(1, 6)]
      + [f"OPIOID_LIQUID_VOL{i}" for i in range(1, 6)])
KEEP = DEMO + PROC + CONC + COM + OUTC + OP

t = time.time()
d = ps.read_year(2024, cpts=CPTS, keep=KEEP)
d = ps.normalize_strings(d)
d.to_pickle(os.path.join(DATA, "cohort_raw.pkl"))
print(f"sports 2024 cohort: {len(d):,} rows x {d.shape[1]} cols in {time.time()-t:.0f}s")
print("\nby CPT:\n", d["CPT"].value_counts().to_string())
print("\nOPIOID_DISCHARGE:\n", d["OPIOID_DISCHARGE"].astype(str).str.strip().value_counts(dropna=False).to_string())
knee = d["CPT"].isin(KNEE); sh = d["CPT"].isin(SHOULDER)
print(f"\nknee pool n={int(knee.sum()):,}  shoulder pool n={int(sh.sum()):,}")
