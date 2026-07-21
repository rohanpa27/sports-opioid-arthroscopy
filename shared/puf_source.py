"""
puf_source.py  -  unified ACS-NSQIP PUF reader for the 2010-2024 replication series.

One function, read_year(year, cpts, keep), returns an upper-cased, string-typed,
CPT-filtered DataFrame for any year 2010-2024, transparently resolving the best
on-disk source and extracting/caching as needed. Output is byte-compatible with the
tab-delimited _aff_work TXT contract every existing study cohort builder assumes:
every value is a stripped string, numeric fields render without float noise, and the
-99 lab sentinel is preserved as the string "-99" (which the cohort _num() masks).

Source tiers (resolved automatically):
  2016-2021  -> already-extracted TXT in _aff_work/            (read directly)
  2012-2015  -> TXT self-extracting .exe in Data Files/        (carve 7z -> .txt, cache)
  2010,2011,2022-2024 -> SPSS .exe in ACS NSQIP 2010-2024/     (carve 7z -> .sav, read pyreadstat)

Nothing here is NSQIP-year-format-specific downstream: after read_year the frame is
uniform, so cohort/analysis code is identical across TXT and SPSS years.

Design notes (verified empirically, see _shared/reference/VARIABLE_CHANGES.md):
  - 2019 TXT ships UPPERCASE headers; we up-case all headers every year so it is moot.
  - SPSS stores categorical fields (STEROID, ASACLAS, SUPINFEC, Age incl. "90+") as
    STRING variables already matching the TXT labels; only true measurements
    (HEIGHT/WEIGHT/OPTIME/TOTHLOS/PR* labs) come back float64 with -99 stored literally.
    format(v,"g") cleans float noise (83.99999999999999 -> "84") and drops ".0".
  - DYSPNEA, WTLOSS, EMERGNCY, ELECTSURG were retired by NSQIP in 2022+; absent columns
    are simply not returned. Cohort code must use .get() guards (see common.py).
"""
import os
import glob
import io
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
RAW_ROOT  = "/Users/rohan/Documents/NSQIP Raw Data"
ACS       = os.path.join(RAW_ROOT, "ACS NSQIP 2010-2024")
AFF       = os.path.join(RAW_ROOT, "_aff_work")
DATAFILES = os.path.join(RAW_ROOT, "Data Files")
CACHE     = os.path.join(ACS, "_shared", "_cache")
os.makedirs(CACHE, exist_ok=True)

SEVENZ_SIG = b"7z\xbc\xaf\x27\x1c"

# already-extracted tab-delimited TXT (read directly, zero extraction)
TXT_AFF = {
    2016: "acs_nsqip_puf16.txt", 2017: "acs_nsqip_puf17.txt",
    2018: "acs_nsqip_puf18_v2.txt", 2019: "acs_nsqip_puf19.txt",
    2020: "acs_nsqip_puf20.txt", 2021: "acs_nsqip_puf21.txt",
}
# years whose only/best source is a TXT self-extracting .exe in Data Files
TXT_EXE_YEARS = {2012, 2013, 2014, 2015}
# years available only as SPSS (must extract .sav, read via pyreadstat)
SPSS_YEARS = {2010, 2011, 2022, 2023, 2024}

DEFAULT_YEARS = list(range(2012, 2025))   # 2012-2024 (project default)


def _iglob_one(folder, pattern):
    """Case-insensitive single-file glob; raises if not exactly resolvable."""
    hits = []
    low = pattern.lower()
    for f in os.listdir(folder):
        # crude case-insensitive fnmatch on the fixed pieces
        fl = f.lower()
        if fl.startswith(low.split("*")[0]) and all(part in fl for part in low.split("*") if part):
            hits.append(os.path.join(folder, f))
    if not hits:
        raise FileNotFoundError(f"No match for {pattern!r} in {folder}")
    hits.sort()
    return hits[0]


# ---------------------------------------------------------------- 7z carve
def _carve_member(exe_path, want_ext, out_path):
    """Extract the first archive member ending in want_ext from a 7z self-extracting
    .exe to out_path (cached: skipped if out_path already exists). NSQIP .exe files are
    7-Zip SFX archives; the real 7z payload begins at the 7z signature after a PE stub."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    import py7zr
    raw = open(exe_path, "rb").read()
    # try every 7z signature offset (usually one) until py7zr opens it
    start = 0
    last_err = None
    while True:
        off = raw.find(SEVENZ_SIG, start)
        if off == -1:
            break
        try:
            with py7zr.SevenZipFile(io.BytesIO(raw[off:]), "r") as z:
                target = next((n for n in z.getnames() if n.lower().endswith(want_ext)), None)
                if target is None:
                    start = off + 1
                    continue
                tmp = out_path + ".tmp_extract"
                os.makedirs(tmp, exist_ok=True)
                z.extract(path=tmp, targets=[target])
                os.replace(os.path.join(tmp, target), out_path)
                # cleanup tmp tree
                for root, dirs, files in os.walk(tmp, topdown=False):
                    for fn in files:
                        os.remove(os.path.join(root, fn))
                    os.rmdir(root)
                return out_path
        except Exception as e:  # not a valid 7z at this offset; advance
            last_err = e
            start = off + 1
    raise RuntimeError(f"Could not carve *{want_ext} from {exe_path}: {last_err}")


def ensure_txt(year):
    """Return a path to a readable tab-delimited TXT for a TXT-tier year."""
    if year in TXT_AFF:
        p = os.path.join(AFF, TXT_AFF[year])
        if not os.path.exists(p):
            raise FileNotFoundError(f"expected extracted TXT missing: {p}")
        return p
    if year in TXT_EXE_YEARS:
        out = os.path.join(CACHE, f"acs_nsqip_puf{year % 100:02d}.txt")
        exe = _iglob_one(DATAFILES, f"acs_nsqip_puf{year % 100:02d}*txt*.exe")
        return _carve_member(exe, ".txt", out)
    raise ValueError(f"{year} is not a TXT-tier year")


def ensure_sav(year):
    """Return a path to an extracted SPSS .sav for an SPSS-tier year."""
    out = os.path.join(CACHE, f"acs_nsqip_puf{year % 100:02d}.sav")
    exe = _iglob_one(ACS, f"acs_nsqip_puf{year % 100:02d}*spss*.exe")
    return _carve_member(exe, ".sav", out)


# ---------------------------------------------------------------- value cleaning
def _clean_float_series(s):
    """float64 SPSS column -> TXT-style string: NaN->'', 84.0->'84', 4.60000001->'4.6',
    -99.0->'-99' (preserved so downstream _num() masks it)."""
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return format(float(v), "g")
    return s.map(fmt)


def _uniform_strings(df):
    """Coerce a mixed-dtype (SPSS) frame to the all-string TXT contract."""
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
            out[c] = _clean_float_series(s)
        else:
            out[c] = s.astype("string").fillna("").astype(str).str.strip()
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------- readers
def _read_txt(path, cpts, keep, chunksize):
    hdr = pd.read_csv(path, sep="\t", nrows=0, encoding="latin-1")
    cptcol = {c.upper(): c for c in hdr.columns}["CPT"]
    parts = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, encoding="latin-1",
                             chunksize=chunksize, na_filter=False, on_bad_lines="skip"):
        if cpts is not None:
            chunk = chunk[chunk[cptcol].astype(str).str.strip().isin(cpts)]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        d = pd.DataFrame(columns=[c.upper() for c in hdr.columns])
    else:
        d = pd.concat(parts, ignore_index=True)
        d.columns = [c.upper() for c in d.columns]
    if keep is not None:
        d = d[[c for c in keep if c in d.columns]].copy()
    return d


def _read_sav(path, cpts, keep, chunksize):
    import pyreadstat
    # Force LATIN1: some NSQIP .sav files (e.g. 2022) declare an encoding that does not
    # match their bytes, and pyreadstat aborts with "invalid byte sequence". Every byte
    # is valid in latin-1, so this never fails and matches the TXT reader's latin-1 path.
    # PUF string fields are ASCII labels/codes, so no information is lost.
    ENC = "LATIN1"
    _, meta = pyreadstat.read_sav(path, metadataonly=True, encoding=ENC)
    namemap = {c.upper(): c for c in meta.column_names}      # UPPER -> raw
    # only read the columns we need (huge memory saving on the 8.5 GB files)
    wanted_upper = set(keep) if keep is not None else set(namemap)
    wanted_upper.add("CPT")
    usecols_raw = [namemap[u] for u in wanted_upper if u in namemap]
    parts = []
    reader = pyreadstat.read_file_in_chunks(
        pyreadstat.read_sav, path, chunksize=chunksize, usecols=usecols_raw, encoding=ENC)
    for chunk, _m in reader:
        chunk.columns = [c.upper() for c in chunk.columns]
        chunk = _uniform_strings(chunk)
        if cpts is not None:
            chunk = chunk[chunk["CPT"].str.strip().isin(cpts)]
        if len(chunk):
            parts.append(chunk)
    if parts:
        d = pd.concat(parts, ignore_index=True)
    else:
        d = pd.DataFrame(columns=[c for c in (keep or []) ])
    if keep is not None:
        d = d[[c for c in keep if c in d.columns]].copy()
    return d


def read_year(year, cpts=None, keep=None, chunksize=200000):
    """Read one PUF year, CPT-filtered, upper-cased, string-typed, with a YEAR column.

    year     : 2010-2024
    cpts     : iterable of principal-CPT strings to keep (None = all rows)
    keep     : list of UPPER-CASE columns to retain (None = all; absent ones skipped)
    """
    cpts = set(str(c).strip() for c in cpts) if cpts is not None else None
    if year in TXT_AFF or year in TXT_EXE_YEARS:
        d = _read_txt(ensure_txt(year), cpts, keep, chunksize)
    elif year in SPSS_YEARS:
        d = _read_sav(ensure_sav(year), cpts, keep, chunksize)
    else:
        raise ValueError(f"unsupported year {year}")
    d["YEAR"] = year
    return d


def normalize_strings(df):
    """Force every non-YEAR column to plain object str with '' for missing.

    pandas 3.0's default `str` dtype keeps missing as NA THROUGH `.astype(str)` (it does
    not render the literal 'nan' the pandas-2-era study code assumed), so a downstream
    `~v.isin([...,'nan',''])` check treats every NA as a POSITIVE event. Concatenating
    years with different columns (IMMUNO_CAT absent pre-2021; DYSPNEA/WTLOSS/EMERGNCY/
    ELECTSURG absent 2022+) introduces exactly those NAs. Filling missing with '' (the
    same thing the TXT na_filter=False reader produces) makes every string check behave
    identically to the original pandas-2 runs."""
    for c in df.columns:
        if c == "YEAR":
            continue
        s = df[c].astype(object)
        df[c] = s.where(s.notna(), "")
    return df


def build_superset(years, cpts, keep, out_path=None, log=print):
    """Read many years, concatenate, optionally pickle. Returns the combined frame."""
    frames = []
    for yr in years:
        d = read_year(yr, cpts=cpts, keep=keep)
        n = len(d)
        by_cpt = ""
        if cpts is not None and n:
            vc = d["CPT"].str.strip().value_counts()
            by_cpt = "  " + " ".join(f"{k}:{v}" for k, v in vc.items())
        if log:
            log(f"  {yr}: {n:>7,} rows{by_cpt}")
        frames.append(d)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = normalize_strings(combined)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        combined.to_pickle(out_path)
        if log:
            log(f"saved {len(combined):,} rows x {combined.shape[1]} cols -> {out_path}")
    return combined


if __name__ == "__main__":
    import sys
    # quick smoke test: read one CPT across a few representative years, one per tier
    cpts = {"27447"}
    for yr in (2019, 2014, 2024):
        d = read_year(yr, cpts=cpts, keep=["CPT", "AGE", "STEROID", "IMMUNO_CAT", "OPTIME", "PRALBUM"])
        print(f"{yr}: {len(d):,} rows | cols={list(d.columns)} | "
              f"IMMUNO_CAT={'yes' if 'IMMUNO_CAT' in d.columns else 'no'} | "
              f"OPTIME sample={list(d['OPTIME'].head(3))} | PRALBUM sample={list(d['PRALBUM'].head(3))}")
