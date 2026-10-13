"""
add — Adaptive Data Dropout (ADD)
=================================
Paper: "Adaptive Data Dropout: Towards Self-Regulated Learning in Deep Neural
        Networks", Gahir, Patel & Gowda.  (paper/ADD.pdf)

Extends PDD by replacing the fixed dropout schedule with feedback-driven
adaptation, using the change in training accuracy as the signal:

    Delta_t = A_t - A_{t-1}

    Paper variant                 File                        Class                   --mode
    ----------------------------  --------------------------  ----------------------  -------------------
    3.2  Adaptive-alpha           adaptive_alpha.py           SimAnnealDropoutSimple  add_adaptive_alpha
    3.3  Adaptive-T (keep-frac)   adaptive_keep_fraction.py   SimAnnealDropout        add_adaptive_t

Result-file tags (how to identify a run in the saved JSON):
    Adaptive-alpha  ->  "<model>_<threshold>_SA_simple"
    Adaptive-T      ->  "<model>_<threshold>_SA_v3_scalar"
"""
