"""
pdd — Progressive Data Dropout (PDD)
====================================
Paper: "Progressive Data Dropout: An Embarrassingly Simple Approach to Train
        Faster", Shriram M S et al., NeurIPS 2025.  (paper/PDD.pdf)

All three PDD variants live as methods on the single `TrainRevision` class in
`pdd_training_variants.py`:

    Paper variant                              Method                       --mode
    -----------------------------------------  ---------------------------  -----------------
    V1  DBPD  (Difficulty-Based Prog. Dropout)  train_with_revision          pdd_dbpd
    V2  SRD   (Scalar Random Dropout)           train_with_percentage        pdd_srd
    V3  SMRD  (Schedule-Matched Random Dropout) train_with_random            pdd_smrd

Appendix C mathematical approximations of the SMRD schedule:

    (1) Power-Law Decay          train_with_power_law           pdd_approx_power_law
    (2) Exponential Decay        train_with_exponential         pdd_approx_exponential
    (3) Logarithmic Decay        train_with_log                 pdd_approx_log
    (4) Inverse Linear Decay     train_with_inverse_linear      pdd_approx_inv_lin
    (5) Sigmoid-Complement Decay train_with_sigmoid_complement  pdd_approx_sigmoid

Plus two pre-paper prototypes (no revision phase, not reported in the paper):
    train_selective        pdd_selective_gradient
    train_selective_epoch  pdd_selective_epoch
And a 3D/video port of DBPD:
    train_with_revision_3d pdd_dbpd_3d
"""
