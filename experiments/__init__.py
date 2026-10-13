"""
experiments — exploratory variants NOT reported in either paper
==============================================================
Everything here is beyond the scope of PDD (paper/PDD.pdf) and ADD
(paper/ADD.pdf). Kept for reference and follow-up work.

    File                  Class            What it is                            --mode
    --------------------  ---------------  ------------------------------------  --------------------
    echo_srd.py           EchoSRD          Oscillating SRD: two decay curves      exp_echo_srd
                                           (alpha_odd / alpha_even) alternating
                                           on odd/even epochs + keep-floor.
                                           Deterministic - NO feedback signal,
                                           so this is NOT an ADD variant.
    genetic_dropout.py    GaDropout        GA over per-sample survival scores.    exp_genetic
    genetic_revision.py   GeneticRevision  Earlier GA scheduler (needs a          exp_genetic_legacy
                                           train/val split).
    simple_switcher.py    SimpleSwitcher   Periodically switches between          exp_switcher
                                           dropout strategies mid-run.
"""
