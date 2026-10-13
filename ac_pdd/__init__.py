"""
ac_pdd — AC-PDD: Adaptive-Composition Progressive Data Dropout
==============================================================
Source: the AC_PDD working repo (Morgan Jones), built on the same original
DBPD/SMRD codebase as pdd/. Not in either paper/PDD.pdf or paper/ADD.pdf.

Where PDD forces a choice between selecting the HARD samples (DBPD) and
selecting a RANDOM subset of the same size (SMRD), AC-PDD keeps DBPD's budget
and schedules the budget's COMPOSITION between those two extremes:

    target_budget      = hard_available                      (DBPD's budget)
    target_hard_frac   = (1 - lambda_t) + lambda_t * (hard_available / batch)

    lambda_t = 0  ->  every slot hard                  == DBPD
    lambda_t = 1  ->  slots in the batch's own ratio   == SMRD in expectation

with lambda_t ramped over the selection phase by utils.get_lambda_t:

    lambda_t = lambda_min + (lambda_max - lambda_min) * progress ** gamma

    Variant                     File             Class.method                --mode
    --------------------------  ---------------  --------------------------  ------------------
    v1  ratio-DBPD              ratio_dbpd.py    RatioDBPD.ratio_dbpd        acpdd_ratio
    v2  ratio-DBPD + budget     ratio_dbpd.py    RatioDBPD.ratio_dbpd_floor  acpdd_ratio_floor
        floor

v2 bounds the budget below at `floor_fraction * batch_size`, so the selection
pool cannot collapse to a tiny residual late in a long run the way DBPD and v1
both do; the shortfall is filled by a uniform random draw from the batch's easy
pool. The two are otherwise identical, kept separate so the un-floored version
remains a clean comparison condition.

Since lambda_min == lambda_max == 0 reproduces DBPD and lambda_min ==
lambda_max == 1 reproduces SMRD, both PDD baselines are reachable as special
cases of the same method.

NOT SHARED WITH pdd/
--------------------
AC-PDD runs on its own `RatioDBPD` class, not on pdd.TrainRevision, because it
adds MetricsRecorder instrumentation and checkpoint/resume that the PDD class
does not have. pdd/, add/ and experiments/ are untouched by this package.

    learning_cycles.py   MetricsRecorder / IndexedDataset / per_sample_loss /
                         make_run_id / set_seed. Needs pandas + pyarrow, and
                         is imported lazily by main.py so that dependency can
                         only ever affect the two acpdd_* modes.

Result files (per run, under --save_path):
    _epochs.parquet  _samples.parquet  _identifiers.parquet  _outcomes.parquet
    _run_metadata.json  _dynamics.png  _ratio_dbpd[_floor]_epoch_stats.csv
    _checkpoint.pt (deleted on clean completion)
    _test / _test.png  <- the legacy JSON+plot collect_results.py reads
"""
