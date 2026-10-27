# Signal Hunters: ML Challenge Hack4Hack 2026, Round 2

Classifies 6-panel ON/OFF radio cadences into 9 classes (eti_candidate, frb,
pulsar, scintillating_star, satellite_pass, radar_chirp, rfi_intermittent,
rfi_persistent, noise). Train = GBT, test = PKS, so the pipeline is built to
handle the telescope shift.

## Approach
1. Preprocessing: remove row glitches and smooth bandpass, normalise per panel,
   mask GBT centre spike.
2. Per-panel drift search over a grid of drift rates (feats.py).
3. Cadence logic: fine-grained line search over global time using
   slew_gap_rows. It checks that ON panels lie on one track and OFF panels
   are empty (feats2.py).
4. Broadband features: simple dedispersion, channel coherence, patchiness.
5. HistGradientBoosting ensemble, class-balanced.
6. Domain shift: drop the most telescope-dependent features.
7. v2: self-training on confident test predictions (final2.py).

## How to reproduce
Put the unzipped dataset in `signal-hunters-round2/`, then run:

    python run_feats.py     # per-panel features
    python run_feats2.py    # cadence track features (~10 min on 1 CPU)
    python final.py         # submission v1
    python final2.py        # submission v2 (self-training)

Only the provided data is used. No outside labels and no hand-labelling.
