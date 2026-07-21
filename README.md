# Discharge Opioid Prescribing After Knee and Shoulder Arthroscopy (ACS-NSQIP 2024)

Analysis code for a cross-sectional study of discharge opioid prescribing after knee and
shoulder arthroscopy, using the discharge-opioid module introduced in the 2024 American
College of Surgeons National Surgical Quality Improvement Program (ACS-NSQIP) Participant
Use File (PUF). This repository is provided for methodological transparency and
reproducibility of the statistical analysis.

## What is here

```
code/
  extract_cohort.py   Carves the 2024 knee and shoulder arthroscopy cohort (by principal
                      CPT code) from the raw 2024 PUF, keeping the demographic, operative,
                      comorbidity, concomitant-procedure, and discharge-opioid fields.
  mme.py              Morphine-milligram-equivalent (MME) conversion engine for the NSQIP
                      discharge-opioid module (CDC/CMS oral conversion factors; per-day
                      and total-course MME across up to five agent slots).
  cohort.py           Single source of truth for the analytic cohort: load() applies the
                      inclusion and exclusion criteria and derives every analysis variable
                      (anatomic pool, procedure labels, bundle burden, covariates, MME
                      outcomes, plausibility filter); strobe() reports the selection ledger.
  analysis.py         All statistical analyses: cohort characteristics, procedure-specific
                      prescribing benchmarks, the concomitant-bundle dose model, the
                      race/ethnicity and sex equity models (modified Poisson risk ratios
                      with E-values, logistic sensitivity), differential MME-computability
                      checks, within-procedure exploratory models with Benjamini-Hochberg
                      correction, and the cross-validated variance-explanation model.
                      Writes the manuscript tables (CSV) and results.json.
shared/
  puf_source.py       Unified ACS-NSQIP PUF reader used by extract_cohort.py. Resolves the
                      on-disk source for a given PUF year, reads it in chunks, up-cases
                      headers, and normalizes SPSS numeric formatting to match the
                      tab-delimited PUF releases.
```

Figure-generation and manuscript-generation code is intentionally not included; this
repository contains the analysis pipeline only.

## Data availability

No data are included in this repository, and none will be shared here. The ACS-NSQIP
Participant Use File is available to participating institutions from the American College
of Surgeons under a data use agreement (https://www.facs.org/quality-programs/data-and-registries/acs-nsqip/participant-use-data-file/)
and may not be redistributed. To reproduce the analysis you must obtain the 2024 PUF
yourself and point the reader at your local copy.

## Reproducing the analysis

1. Obtain the 2024 ACS-NSQIP PUF (SPSS release) under your institution's data use agreement.
2. Edit the paths at the top of `shared/puf_source.py` (and the `sys.path` line in
   `code/extract_cohort.py`) to match where your PUF lives. These paths are machine
   specific by design because the data cannot ship with the code.
3. Run `code/extract_cohort.py` to build `data/cohort_raw.pkl` (reads the roughly 2 GB
   .sav file once and caches the CPT-filtered cohort).
4. Run `code/analysis.py` to produce `outputs/tables/*.csv` and `outputs/results.json`,
   which contains every number reported in the manuscript.

Environment used for the published results: Python 3.14.6, numpy 2.5.1, pandas 3.0.3,
scipy 1.18.0, statsmodels 0.14.6, scikit-learn 1.9.0, pyreadstat for SPSS reading.

## Key design decisions encoded in the pipeline

- Total-course MME is computed as per-dose strength x quantity dispensed x CDC/CMS oral
  conversion factor, summed across agent slots; non-mg routes and unmapped agents are not
  counted (documented limitation).
- A plausibility filter treats a total course above 1500 MME or a labeled daily dose above
  150 MME as a data-entry error (set non-computable, counted and reported).
- The equity model uses modified Poisson regression (robust standard errors) for risk
  ratios as the primary effect measure; work RVU is intentionally omitted from that model
  because it is nearly collinear with the procedure fixed effects, and operative time
  carries the case-complexity signal instead.
- Patients with unreported race are retained as an explicit stratum but reported as a
  documentation-quality signal, not an equity contrast.

## Disclaimer

The American College of Surgeons National Surgical Quality Improvement Program and the
hospitals participating in it are the source of the data used herein; they have not
verified and are not responsible for the statistical validity of the data analysis or the
conclusions derived by the authors.

## Citation

A manuscript describing this study is in preparation. Citation details will be added here
upon publication.
