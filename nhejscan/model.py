#!/usr/bin/env python3
"""
Dynamic DSBR Pathway Classification Model v4
=============================================
Classifies double-strand break repair (DSBR) pathway utilization
across fungal clades from indel size-class count data.

Input data scope
---------------- 
Events in this dataset are restricted to DSB-derived indels. HR-mediated
repair events are excluded at the upstream filtering step and do not
contribute to the counts modelled here.

Pathways modelled
-----------------
  c-NHEJ  : classical non-homologous end joining
              Signature: short insertions (0-1 bp fill-in) and blunt
              deletions (0-1 bp). Mechanistically characterised by
              Ku70/80-dependent end protection and ligation without
              extensive resection.

  alt-EJ  : alternative / microhomology-mediated end joining (MMEJ),
              encompassing theta-mediated EJ and, at longer deletion
              sizes (>=10 bp), SSA-like events that depend on flanking
              microhomology. SSA is treated here as a sub-class of
              alt-EJ because its sequence outcome (deletion to flanking
              repeats) is mechanistically continuous with MMEJ at large
              deletion sizes and cannot be distinguished by size class
              alone without explicit repeat annotation. This is a
              deliberate modelling choice, not an omission.

CAVEATS
-------
1. SSA is intentionally grouped within alt-EJ. The DEL_10_plus bin in
   particular captures both canonical MMEJ and SSA. The alt-EJ fraction
   should therefore be interpreted as 'MMEJ + SSA utilization'. If
   repeat-flanked deletions can be annotated separately in future, SSA
   could be split into a third pathway.
2. Insertions >= 5 bp are mechanistically ambiguous (snap-back synthesis,
   template switching); the 'no_large_ins' variant excludes these.
3. Bootstrap CIs reflect multinomial sampling variance only.
   Weight-perturbation bootstrap propagates prior uncertainty separately.
   Combined uncertainty is reported as the root-sum-of-squares of the two.
4. Small-count clades (total_events < MIN_RELIABLE_EVENTS) are flagged
   as unreliable regardless of CI width.
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr

# ============================================================
# USER SETTINGS
# ============================================================
INPUT_FILE = os.environ.get("NHEJSCAN_MODEL_INPUT", "indel-mh_count_for_model.csv")
OUTPUT_DIR = os.environ.get("NHEJSCAN_MODEL_OUTDIR", "dynamic_pathway_model_output_v4")
N_BOOT = 2000           # bootstrap replicates for multinomial resampling
N_WEIGHT_BOOT = 1000    # replicates for weight-perturbation sensitivity
RANDOM_SEED = int(os.environ.get("NHEJSCAN_MODEL_SEED", "42"))
KEEP_ZERO_EVENT_CLADES = True
ERROR_ON_NEGATIVE_COUNTS = True

# Clades with fewer total events than this are flagged as low-confidence.
# Justify this threshold based on your expected power requirements.
MIN_RELIABLE_EVENTS = 50

# Weight perturbation SD (fraction of the weight value).
# 0.10 = weights are perturbed by +/-~10% of their value (Gaussian).
WEIGHT_PERTURBATION_SD = 0.10

# Sensitivity flag thresholds (fraction units, 0-1).
# These are configurable here rather than buried in the code.
ROBUST_THRESHOLD = 0.05     # max range_across_modes AND variant_range
SENSITIVE_THRESHOLD = 0.15  # min range_across_modes OR variant_range to flag "sensitive"

# ============================================================
# COLUMN MAPPING
# ============================================================
COLS = {
    "clade":       "clade",
    "INS_0_1":     "INS_0_1",
    "INS_2_4":     "INS_2_4",
    "INS_5_9":     "INS_5_9",
    "INS_10_plus": "INS_10_plus",
    "DEL_0_1":     "DEL_0_1",
    "DEL_2_4":     "DEL_2_4",
    "DEL_5_9":     "DEL_5_9",
    "DEL_10_plus": "DEL_10_plus",
}

EVENT_ORDER = [
    "INS_0_1", "INS_2_4", "INS_5_9", "INS_10_plus",
    "DEL_0_1", "DEL_2_4", "DEL_5_9", "DEL_10_plus",
]
INS_EVENTS = ["INS_0_1", "INS_2_4", "INS_5_9", "INS_10_plus"]
DEL_EVENTS = ["DEL_0_1", "DEL_2_4", "DEL_5_9", "DEL_10_plus"]

# ============================================================
# WEIGHT MODES
# Weights = P(event class | c-NHEJ).
# alt-EJ weight = 1 - weight.
# Each weight has a Gaussian perturbation applied in the weight-boot.
#
# Mechanistic justification per bin:
#   INS_0_1    : 1-bp fill-in insertions are the canonical c-NHEJ product
#                (Ku-dependent end ligation after minimal processing);
#                high c-NHEJ prior.
#   INS_2_4    : Short insertions consistent with limited snap-back
#                synthesis at c-NHEJ junctions; moderate-high c-NHEJ prior.
#   INS_5_9    : Mechanistically ambiguous -- could reflect template-switching
#                or extended snap-back; intermediate prior, excluded in
#                'no_large_ins' variant.
#   INS_10_plus: Very rare at c-NHEJ junctions; low c-NHEJ prior.
#   DEL_0_1    : Blunt-end ligation or minimal nucleolytic trimming;
#                consistent with c-NHEJ, moderate prior.
#   DEL_2_4    : Overlaps short-MH MMEJ; ambiguous, intermediate prior.
#   DEL_5_9    : Consistent with MH-mediated alt-EJ (MMEJ); low c-NHEJ prior.
#   DEL_10_plus: Encompasses MMEJ at longer MH tracts and SSA-like events
#                (deletion to flanking direct repeats). SSA is grouped here
#                within alt-EJ because both require resection and annealing
#                to flanking homology, and are indistinguishable by size
#                class alone. Very low c-NHEJ prior.
# ============================================================
WEIGHT_MODES: Dict[str, Dict[str, float]] = {
    "conservative": {
        "INS_0_1": 0.85, "INS_2_4": 0.70, "INS_5_9": 0.45, "INS_10_plus": 0.25,
        "DEL_0_1": 0.65, "DEL_2_4": 0.50, "DEL_5_9": 0.30, "DEL_10_plus": 0.10,
    },
    "moderate": {
        "INS_0_1": 0.95, "INS_2_4": 0.80, "INS_5_9": 0.35, "INS_10_plus": 0.10,
        "DEL_0_1": 0.75, "DEL_2_4": 0.55, "DEL_5_9": 0.20, "DEL_10_plus": 0.05,
    },
    "aggressive": {
        "INS_0_1": 0.99, "INS_2_4": 0.90, "INS_5_9": 0.20, "INS_10_plus": 0.01,
        "DEL_0_1": 0.85, "DEL_2_4": 0.60, "DEL_5_9": 0.10, "DEL_10_plus": 0.01,
    },
}

# ============================================================
# MODEL VARIANTS
# Five variants test sensitivity to inclusion/exclusion of specific size classes:
#   full     : all eight size classes; primary analysis.
#   stable   : excludes INS_5_9 and DEL_5_9, the most mechanistically
#              ambiguous intermediate classes (snap-back / template switching
#              for insertions; overlap between NHEJ trimming and early MMEJ
#              resection for deletions). This is the most defensible
#              conservative primary variant.
#   minimal  : 0-1 bp and >=10 bp classes only. These are the two most
#              diagnostically polar classes: 0-1 bp strongly favours c-NHEJ;
#              >=10 bp strongly favours alt-EJ/MMEJ+SSA. Tests whether the
#              extreme ends of the spectrum drive the classification.
#   ins_only : insertion classes only. Insertions arise almost exclusively
#              from fill-in synthesis at c-NHEJ junctions (short) or from
#              template switching / MMEJ insertions (long). Useful for
#              checking whether the INS spectrum alone supports the model.
#   del_only : deletion classes only. Deletions are the dominant product of
#              both c-NHEJ (blunt trimming) and alt-EJ/MMEJ (resection +
#              annealing). Most mechanistically informative sub-spectrum.
# ============================================================
MODEL_VARIANTS: Dict[str, List[str]] = {
    "full":     EVENT_ORDER,
    "stable":   [e for e in EVENT_ORDER if e not in {"INS_5_9", "DEL_5_9"}],
    "minimal":  ["INS_0_1", "DEL_10_plus"],
    "ins_only": INS_EVENTS,
    "del_only": DEL_EVENTS,
}

# ============================================================
# SETUP
# ============================================================
Path(OUTPUT_DIR).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{OUTPUT_DIR}/run.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
# PROVENANCE
# ============================================================
def write_provenance() -> None:
    prov = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "input_file": INPUT_FILE,
        "n_boot": N_BOOT,
        "n_weight_boot": N_WEIGHT_BOOT,
        "random_seed": RANDOM_SEED,
        "min_reliable_events": MIN_RELIABLE_EVENTS,
        "weight_perturbation_sd": WEIGHT_PERTURBATION_SD,
        "robust_threshold": ROBUST_THRESHOLD,
        "sensitive_threshold": SENSITIVE_THRESHOLD,
        "weight_modes": WEIGHT_MODES,
        "model_variants": {k: v for k, v in MODEL_VARIANTS.items()},
        "modelling_assumptions": {
            "HR_excluded": "HR events removed at upstream filtering; not present in input counts.",
            "SSA_treatment": "SSA grouped within alt-EJ. DEL_10_plus captures MMEJ and SSA-like deletions to flanking direct repeats, indistinguishable by size class alone without repeat annotation.",
            "altEJ_label_scope": "alt-EJ fraction = MMEJ + SSA utilization.",
        },
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    with open(f"{OUTPUT_DIR}/provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)
    log.info("Provenance written to %s/provenance.json", OUTPUT_DIR)

# ============================================================
# HELPERS
# ============================================================
def safe_divide(num: float, den: float) -> float:
    if den == 0 or pd.isna(den) or pd.isna(num):
        return np.nan
    return num / den


def validate_weights(weight_modes: Dict[str, Dict[str, float]]) -> None:
    for mode_name, weights in weight_modes.items():
        missing = [e for e in EVENT_ORDER if e not in weights]
        if missing:
            raise ValueError(f"Weight mode '{mode_name}' missing events: {missing}")
        for e, w in weights.items():
            if not (0.0 <= w <= 1.0):
                raise ValueError(
                    f"Weight mode '{mode_name}', event '{e}': weight {w} outside [0,1]."
                )


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    required = [COLS["clade"]] + [COLS[e] for e in EVENT_ORDER]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[COLS["clade"]] = df[COLS["clade"]].astype(str).str.strip()

    if (df[COLS["clade"]] == "").any():
        raise ValueError("Empty clade names found.")
    if df[COLS["clade"]].duplicated().any():
        dupes = df.loc[df[COLS["clade"]].duplicated(), COLS["clade"]].tolist()
        raise ValueError(f"Duplicate clade names: {dupes}")

    for e in EVENT_ORDER:
        df[COLS[e]] = pd.to_numeric(df[COLS[e]], errors="coerce")

    bad_numeric = df[[COLS[e] for e in EVENT_ORDER]].isna().any(axis=1)
    if bad_numeric.any():
        bad_clades = df.loc[bad_numeric, COLS["clade"]].tolist()
        raise ValueError(f"Non-numeric/missing counts in clades: {bad_clades}")

    if ERROR_ON_NEGATIVE_COUNTS:
        neg_mask = (df[[COLS[e] for e in EVENT_ORDER]] < 0).any(axis=1)
        if neg_mask.any():
            bad_clades = df.loc[neg_mask, COLS["clade"]].tolist()
            raise ValueError(f"Negative counts in clades: {bad_clades}")

    frac_mask = (df[[COLS[e] for e in EVENT_ORDER]] % 1 != 0).any(axis=1)
    if frac_mask.any():
        warnings.warn(
            "Non-integer event counts detected. Bootstrap resampling assumes "
            "count data; results may be unreliable.",
            stacklevel=2,
        )

    df["total_events_raw"] = df[[COLS[e] for e in EVENT_ORDER]].sum(axis=1)
    low_n = df["total_events_raw"] < MIN_RELIABLE_EVENTS
    if low_n.any():
        low_clades = df.loc[low_n, COLS["clade"]].tolist()
        log.warning(
            "Clades below MIN_RELIABLE_EVENTS (%d): %s -- flagged as low_confidence.",
            MIN_RELIABLE_EVENTS, low_clades,
        )

    if not KEEP_ZERO_EVENT_CLADES:
        df = df[df["total_events_raw"] > 0].copy()

    return df


# ============================================================
# POINT-ESTIMATE SCORING
# ============================================================
def compute_scores_for_mode(
    df: pd.DataFrame,
    mode_name: str,
    weights: Dict[str, float],
    variant_name: str,
    selected_events: List[str],
) -> pd.DataFrame:
    """
    Compute point-estimate cNHEJ/altEJ fractions for each clade.
    Per-event contributions are recorded for downstream driver analysis.
    """
    out_rows = []

    for _, row in df.iterrows():
        clade = row[COLS["clade"]]
        counts = {e: float(row[COLS[e]]) for e in selected_events}
        total_events = sum(counts.values())

        c_contrib = {e: counts[e] * weights[e] for e in selected_events}
        a_contrib = {e: counts[e] * (1.0 - weights[e]) for e in selected_events}

        c_score = sum(c_contrib.values())
        a_score = sum(a_contrib.values())
        total_score = c_score + a_score

        c_frac = safe_divide(c_score, total_score)
        a_frac = safe_divide(a_score, total_score)

        c_ins = sum(c_contrib[e] for e in selected_events if e in INS_EVENTS)
        c_del = sum(c_contrib[e] for e in selected_events if e in DEL_EVENTS)
        a_ins = sum(a_contrib[e] for e in selected_events if e in INS_EVENTS)
        a_del = sum(a_contrib[e] for e in selected_events if e in DEL_EVENTS)
        c_route = c_ins + c_del
        a_route = a_ins + a_del

        row_out: Dict = {
            "clade":               clade,
            "mode":                mode_name,
            "variant":             variant_name,
            "n_selected_classes":  len(selected_events),
            "selected_events":     ",".join(selected_events),
            "total_events":        total_events,
            "low_confidence":      total_events < MIN_RELIABLE_EVENTS,
            "cNHEJ_score":         c_score,
            "altEJ_score":         a_score,
            "cNHEJ_fraction":      c_frac,
            "altEJ_fraction":      a_frac,
            "cNHEJ_INS_fraction":  safe_divide(c_ins, c_route),
            "cNHEJ_DEL_fraction":  safe_divide(c_del, c_route),
            "altEJ_INS_fraction":  safe_divide(a_ins, a_route),
            "altEJ_DEL_fraction":  safe_divide(a_del, a_route),
            "raw_INS_total":       sum(counts[e] for e in selected_events if e in INS_EVENTS),
            "raw_DEL_total":       sum(counts[e] for e in selected_events if e in DEL_EVENTS),
        }

        for e in EVENT_ORDER:
            if e in selected_events:
                row_out[f"count_{e}"]     = counts[e]
                row_out[f"c_contrib_{e}"] = c_contrib[e]
                row_out[f"a_contrib_{e}"] = a_contrib[e]
                row_out[f"raw_fraction_{e}"] = safe_divide(counts[e], total_events)
                row_out[f"c_score_frac_{e}"] = safe_divide(c_contrib[e], c_score)
                row_out[f"a_score_frac_{e}"] = safe_divide(a_contrib[e], a_score)
            else:
                row_out[f"count_{e}"]        = np.nan
                row_out[f"c_contrib_{e}"]    = np.nan
                row_out[f"a_contrib_{e}"]    = np.nan
                row_out[f"raw_fraction_{e}"] = np.nan
                row_out[f"c_score_frac_{e}"] = np.nan
                row_out[f"a_score_frac_{e}"] = np.nan

        if total_events > 0 and c_score > 0:
            row_out["top_cNHEJ_driver"] = max(selected_events, key=lambda x: c_contrib[x])
        else:
            row_out["top_cNHEJ_driver"] = np.nan

        if total_events > 0 and a_score > 0:
            row_out["top_altEJ_driver"] = max(selected_events, key=lambda x: a_contrib[x])
        else:
            row_out["top_altEJ_driver"] = np.nan

        out_rows.append(row_out)

    return pd.DataFrame(out_rows)


# ============================================================
# BOOTSTRAP 1: MULTINOMIAL SAMPLING UNCERTAINTY
# ============================================================
def _bootstrap_multinomial(
    counts: np.ndarray,
    weights_arr: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample event counts as multinomial draws, holding weights fixed.
    Returns arrays of cNHEJ and altEJ fractions over replicates.
    """
    total = int(np.round(counts.sum()))
    probs = counts / counts.sum()
    c_fracs, a_fracs = [], []
    for _ in range(n_boot):
        boot_counts = rng.multinomial(total, probs).astype(float)
        c = np.dot(boot_counts, weights_arr)
        a = np.dot(boot_counts, 1.0 - weights_arr)
        denom = c + a
        c_fracs.append(c / denom if denom > 0 else np.nan)
        a_fracs.append(a / denom if denom > 0 else np.nan)
    return np.array(c_fracs, dtype=float), np.array(a_fracs, dtype=float)


def bootstrap_multinomial_all(
    df: pd.DataFrame,
    mode_name: str,
    weights: Dict[str, float],
    variant_name: str,
    selected_events: List[str],
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rows = []
    weights_arr = np.array([weights[e] for e in selected_events])
    for idx, row in df.iterrows():
        counts = np.array([float(row[COLS[e]]) for e in selected_events])
        total = counts.sum()
        rng = np.random.default_rng(seed + idx)
        if total <= 0:
            res = {k: np.nan for k in [
                "cNHEJ_fraction_mean", "cNHEJ_fraction_ci_low", "cNHEJ_fraction_ci_high",
                "altEJ_fraction_mean",  "altEJ_fraction_ci_low",  "altEJ_fraction_ci_high",
                "cNHEJ_fraction_sd", "altEJ_fraction_sd",
            ]}
        else:
            c_fracs, a_fracs = _bootstrap_multinomial(counts, weights_arr, n_boot, rng)
            res = {
                "cNHEJ_fraction_mean":    np.nanmean(c_fracs),
                "cNHEJ_fraction_ci_low":  np.nanpercentile(c_fracs, 2.5),
                "cNHEJ_fraction_ci_high": np.nanpercentile(c_fracs, 97.5),
                "altEJ_fraction_mean":    np.nanmean(a_fracs),
                "altEJ_fraction_ci_low":  np.nanpercentile(a_fracs, 2.5),
                "altEJ_fraction_ci_high": np.nanpercentile(a_fracs, 97.5),
                "cNHEJ_fraction_sd":      np.nanstd(c_fracs, ddof=1),
                "altEJ_fraction_sd":      np.nanstd(a_fracs, ddof=1),
            }
        res.update({"clade": row[COLS["clade"]], "mode": mode_name, "variant": variant_name})
        rows.append(res)
    return pd.DataFrame(rows)


# ============================================================
# BOOTSTRAP 2: WEIGHT-PERTURBATION UNCERTAINTY
# ============================================================
def weight_perturbation_boot(
    df: pd.DataFrame,
    mode_name: str,
    weights: Dict[str, float],
    variant_name: str,
    selected_events: List[str],
    n_boot: int = N_WEIGHT_BOOT,
    perturbation_sd: float = WEIGHT_PERTURBATION_SD,
    seed: int = RANDOM_SEED + 99999,
) -> pd.DataFrame:
    """
    Assess sensitivity to uncertainty in the prior weights.
    Each replicate draws perturbed weights from a truncated Gaussian
    centered on the nominal weight with SD = perturbation_sd * weight,
    then clips to [0.01, 0.99] to avoid degenerate cases.

    This propagates model prior uncertainty separately from sampling
    uncertainty. Combined SD is reported as RSS in make_run_summary.
    """
    rng = np.random.default_rng(seed)
    nominal = np.array([weights[e] for e in selected_events])
    rows = []

    for _, row in df.iterrows():
        clade = row[COLS["clade"]]
        counts = np.array([float(row[COLS[e]]) for e in selected_events])
        total = counts.sum()
        if total <= 0:
            rows.append({
                "clade": clade, "mode": mode_name, "variant": variant_name,
                "weight_boot_cNHEJ_sd": np.nan,
                "weight_boot_cNHEJ_ci_low": np.nan,
                "weight_boot_cNHEJ_ci_high": np.nan,
            })
            continue

        c_fracs = []
        for _ in range(n_boot):
            perturbed = rng.normal(nominal, perturbation_sd * nominal)
            perturbed = np.clip(perturbed, 0.01, 0.99)
            c = np.dot(counts, perturbed)
            a = np.dot(counts, 1.0 - perturbed)
            denom = c + a
            c_fracs.append(c / denom if denom > 0 else np.nan)

        c_arr = np.array(c_fracs, dtype=float)
        rows.append({
            "clade": clade, "mode": mode_name, "variant": variant_name,
            "weight_boot_cNHEJ_sd":       np.nanstd(c_arr, ddof=1),
            "weight_boot_cNHEJ_ci_low":   np.nanpercentile(c_arr, 2.5),
            "weight_boot_cNHEJ_ci_high":  np.nanpercentile(c_arr, 97.5),
        })

    return pd.DataFrame(rows)


# ============================================================
# RANK STABILITY WITH RANK CORRELATION
# ============================================================
def compare_rank_stability(
    results_df: pd.DataFrame, variant_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare clade rankings across weighting modes using Kendall's tau
    and Spearman's rho. Returns:
      pivot_df  : cNHEJ_fraction wide table (clade x mode)
      rank_df   : ranks and rank range per clade
      corr_df   : pairwise rank correlation between modes
    """
    sub = results_df[results_df["variant"] == variant_name].copy()
    pivot = sub.pivot(index="clade", columns="mode", values="cNHEJ_fraction").reset_index()

    rank_df = pivot.copy()
    mode_cols = [c for c in pivot.columns if c != "clade"]
    for mode in mode_cols:
        rank_df[f"{mode}_rank"] = rank_df[mode].rank(ascending=False, method="average")

    rank_cols = [f"{m}_rank" for m in mode_cols]
    rank_df["rank_range"] = rank_df[rank_cols].max(axis=1) - rank_df[rank_cols].min(axis=1)

    # Pairwise rank correlations
    corr_rows = []
    for i, m1 in enumerate(mode_cols):
        for m2 in mode_cols[i + 1:]:
            x = pivot[m1].values
            y = pivot[m2].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 3:
                tau, p_tau = kendalltau(x[mask], y[mask])
                rho, p_rho = spearmanr(x[mask], y[mask])
            else:
                tau = p_tau = rho = p_rho = np.nan
            corr_rows.append({
                "mode_A": m1, "mode_B": m2,
                "kendall_tau": tau, "kendall_p": p_tau,
                "spearman_rho": rho, "spearman_p": p_rho,
                "n_clades": int(mask.sum()),
            })
    corr_df = pd.DataFrame(corr_rows)

    return pivot, rank_df, corr_df


def compare_variants_within_mode(
    results_df: pd.DataFrame, mode_name: str
) -> pd.DataFrame:
    sub = results_df[results_df["mode"] == mode_name].copy()
    pivot = sub.pivot(index="clade", columns="variant", values="cNHEJ_fraction").reset_index()
    variant_cols = [c for c in pivot.columns if c != "clade"]
    pivot["variant_range"] = (
        pivot[variant_cols].max(axis=1) - pivot[variant_cols].min(axis=1)
        if len(variant_cols) >= 2 else np.nan
    )
    return pivot.sort_values("variant_range", ascending=False)


def mode_sensitivity_table(results_df: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    sub = results_df[results_df["variant"] == variant_name].copy()
    wide = sub.pivot(index="clade", columns="mode", values="cNHEJ_fraction").reset_index()
    mode_cols = [c for c in wide.columns if c != "clade"]
    wide["range_across_modes"] = wide[mode_cols].max(axis=1) - wide[mode_cols].min(axis=1)
    return wide.sort_values("range_across_modes", ascending=False)


def identify_top_drivers(results_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "clade", "mode", "variant", "total_events", "low_confidence",
        "cNHEJ_fraction", "altEJ_fraction",
        "top_cNHEJ_driver", "top_altEJ_driver",
    ]
    return results_df[keep_cols].sort_values(
        ["variant", "mode", "cNHEJ_fraction"], ascending=[True, True, False]
    )


# ============================================================
# COMBINED UNCERTAINTY (RSS)
# ============================================================
def combine_uncertainties(merged_df: pd.DataFrame, wboot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge multinomial bootstrap SD with weight-perturbation SD.
    Combined SD = sqrt(multinomial_SD^2 + weight_boot_SD^2).
    This is an approximation assuming the two sources are independent.
    """
    combined = merged_df.merge(wboot_df, on=["clade", "mode", "variant"], how="left")
    combined["cNHEJ_fraction_combined_sd"] = np.sqrt(
        combined["cNHEJ_fraction_sd"].fillna(0) ** 2
        + combined["weight_boot_cNHEJ_sd"].fillna(0) ** 2
    )
    return combined



# ============================================================
# PAIRWISE BOOTSTRAP SIGNIFICANCE TESTS
# ============================================================
def pairwise_bootstrap_tests(
    df: pd.DataFrame,
    mode_name: str = "moderate",
    variant_name: str = "full",
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Test whether f_cNHEJ differs significantly between every pair of clades.

    Approach
    --------
    For each pair of clades (A, B), re-run the multinomial bootstrap
    independently for both clades using the same weights and variant.
    In each of the n_boot replicates, compute:

        delta_k = f_cNHEJ(A, k) - f_cNHEJ(B, k)

    From this delta distribution:
      delta_mean            : mean difference (approx. f_A - f_B)
      delta_ci_low/high     : 2.5th and 97.5th percentiles of delta
      p_value_1sided        : fraction of replicates where delta <= 0
                              (prob. that B >= A under resampling)
      p_value_2sided        : 2 * min(p_1sided, 1 - p_1sided)
      significant_2sided    : True if p_value_2sided < 0.05

    This test propagates sampling uncertainty only (weights held fixed).
    Compare with mode_sensitivity output for weight-prior robustness.

    The minimum reportable p-value is 1/n_boot.
    """
    weights         = WEIGHT_MODES[mode_name]
    selected_events = MODEL_VARIANTS[variant_name]
    weights_arr     = np.array([weights[e] for e in selected_events])

    # Per-clade bootstrap distributions and point estimates
    clade_fracs: Dict[str, np.ndarray] = {}
    point_estimates: Dict[str, float]  = {}

    for idx, row in df.iterrows():
        clade  = row[COLS["clade"]]
        counts = np.array([float(row[COLS[e]]) for e in selected_events])
        total  = counts.sum()
        if total <= 0:
            clade_fracs[clade]     = np.full(n_boot, np.nan)
            point_estimates[clade] = np.nan
            continue
        c_score = np.dot(counts, weights_arr)
        a_score = np.dot(counts, 1.0 - weights_arr)
        point_estimates[clade] = c_score / (c_score + a_score)
        rng = np.random.default_rng(seed + idx)
        c_fracs, _ = _bootstrap_multinomial(counts, weights_arr, n_boot, rng)
        clade_fracs[clade] = c_fracs

    clades = [
        row[COLS["clade"]] for _, row in df.iterrows()
        if not np.isnan(point_estimates.get(row[COLS["clade"]], np.nan))
    ]

    rows = []
    for i, clade_i in enumerate(clades):
        for clade_j in clades[i + 1:]:
            fi = point_estimates[clade_i]
            fj = point_estimates[clade_j]
            # Ensure A is the higher-f clade using local vars — never mutate loop vars
            if fi >= fj:
                clade_a, clade_b, fa, fb = clade_i, clade_j, fi, fj
            else:
                clade_a, clade_b, fa, fb = clade_j, clade_i, fj, fi

            deltas = clade_fracs[clade_a] - clade_fracs[clade_b]
            deltas = deltas[~np.isnan(deltas)]
            p_min  = 1.0 / len(deltas)

            delta_mean    = float(np.mean(deltas))
            delta_ci_low  = float(np.percentile(deltas, 2.5))
            delta_ci_high = float(np.percentile(deltas, 97.5))
            p_1sided      = max(float(np.mean(deltas <= 0)), p_min)
            p_2sided      = max(float(2 * min(p_1sided, 1.0 - p_1sided)), p_min)

            rows.append({
                "mode":               mode_name,
                "variant":            variant_name,
                "clade_A":            clade_a,
                "clade_B":            clade_b,
                "f_cNHEJ_A":          round(fa, 6),
                "f_cNHEJ_B":          round(fb, 6),
                "delta_mean":         round(delta_mean, 6),
                "delta_ci_low":       round(delta_ci_low, 6),
                "delta_ci_high":      round(delta_ci_high, 6),
                "p_value_1sided":     round(p_1sided, 6),
                "p_value_2sided":     round(p_2sided, 6),
                "significant_1sided": p_1sided < 0.05,
                "significant_2sided": p_2sided < 0.05,
                "n_boot":             len(deltas),
            })

    out = pd.DataFrame(rows).sort_values(["clade_A", "clade_B"]).reset_index(drop=True)

    log.info("Pairwise bootstrap tests (%s / %s):", mode_name, variant_name)
    for _, r in out.iterrows():
        sig = "**" if r["significant_2sided"] else "  "
        log.info(
            "  %s %s vs %s: delta=%.3f [%.3f, %.3f]  p(2-sided)=%.4f",
            sig, r["clade_A"], r["clade_B"],
            r["delta_mean"], r["delta_ci_low"], r["delta_ci_high"],
            r["p_value_2sided"],
        )
    return out

# ============================================================
# RUN SUMMARY
# ============================================================
def make_run_summary(
    results_df: pd.DataFrame,
    combined_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact summary based on moderate/full model.
    Interpretation flags use configurable thresholds.
    Combined uncertainty integrates multinomial and weight-prior variance.
    """
    base = results_df[
        (results_df["mode"] == "moderate") & (results_df["variant"] == "full")
    ].copy()
    if base.empty:
        return pd.DataFrame()

    ci = combined_df[
        (combined_df["mode"] == "moderate") & (combined_df["variant"] == "full")
    ][[
        "clade",
        "cNHEJ_fraction_ci_low", "cNHEJ_fraction_ci_high",
        "cNHEJ_fraction_sd",
        "weight_boot_cNHEJ_sd", "weight_boot_cNHEJ_ci_low", "weight_boot_cNHEJ_ci_high",
        "cNHEJ_fraction_combined_sd",
    ]].copy()

    sens = mode_sensitivity_table(results_df, variant_name="full")[
        ["clade", "range_across_modes"]
    ].copy()

    var_sens = compare_variants_within_mode(results_df, mode_name="moderate")[
        ["clade", "variant_range"]
    ].copy()

    out = (
        base
        .merge(ci, on="clade", how="left")
        .merge(sens, on="clade", how="left")
        .merge(var_sens, on="clade", how="left")
    )

    out["interpretation_flag"] = np.where(
        out["low_confidence"],
        "low_confidence",
        np.where(
            (out["range_across_modes"] <= ROBUST_THRESHOLD)
            & (out["variant_range"] <= ROBUST_THRESHOLD),
            "robust",
            np.where(
                (out["range_across_modes"] > SENSITIVE_THRESHOLD)
                | (out["variant_range"] > SENSITIVE_THRESHOLD),
                "sensitive",
                "intermediate",
            ),
        ),
    )

    keep_cols = [
        "clade", "total_events", "low_confidence",
        "cNHEJ_fraction", "altEJ_fraction",
        "cNHEJ_fraction_ci_low", "cNHEJ_fraction_ci_high",
        "cNHEJ_fraction_sd",
        "weight_boot_cNHEJ_sd", "weight_boot_cNHEJ_ci_low", "weight_boot_cNHEJ_ci_high",
        "cNHEJ_fraction_combined_sd",
        "range_across_modes", "variant_range",
        "top_cNHEJ_driver", "top_altEJ_driver",
        "interpretation_flag",
    ]
    return out[keep_cols].sort_values("cNHEJ_fraction", ascending=False)


# ============================================================
# PLOT STYLE
# All figures share one palette and typography system.
# White background, dark slate text, zero chartjunk.
# Blue family -> c-NHEJ (end protection / Ku-dependent)
# Red family  -> alt-EJ / MMEJ+SSA (end resection / theta-mediated)
# ============================================================
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Liberation Sans", "DejaVu Sans", "FreeSans"],
    "mathtext.fontset":   "custom",
    "mathtext.rm":        "Liberation Sans",
    "mathtext.it":        "Liberation Sans:italic",
    "mathtext.bf":        "Liberation Sans:bold",
    "pdf.fonttype":       42,   # embed TrueType in PDF (not Type 3)
    "svg.fonttype":       "none",
    "font.size":          9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.linewidth":     0.6,
    "axes.labelpad":      5,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "axes.labelcolor":    "#1A1A1A",
    "text.color":         "#1A1A1A",
    "xtick.color":        "#1A1A1A",
    "ytick.color":        "#1A1A1A",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "legend.frameon":     True,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   "#DDDDDD",
    "legend.fontsize":    7.5,
    "legend.handlelength":1.4,
})

C = {
    # c-NHEJ palette -- steel blue family
    "nhej_dark":    "#14375A",
    "nhej_mid":     "#1F6FA8",
    "nhej_light":   "#7BB8D9",
    "nhej_pale":    "#C8E3F2",
    # alt-EJ palette -- warm red family
    "altej_dark":   "#7A1A1A",
    "altej_mid":    "#C0392B",
    "altej_light":  "#E8836E",
    "altej_pale":   "#F5C9C0",
    # chrome
    "bg":           "#FFFFFF",
    "bg_panel":     "#FAFAFA",
    "grid":         "#E8E8E8",
    "rule":         "#CCCCCC",
    "text":         "#1A1A1A",
    "subtext":      "#666666",
    # robustness flags -- accessible palette
    "robust":       "#1A6B34",
    "intermediate": "#A05C10",
    "sensitive":    "#B52020",
    "low_conf":     "#999999",
}

DONUT = {
    # Outer ring -- exactly as in the original static figure
    "outer_nhej":     "plum",           # c-NHEJ (end protection)
    "outer_altej":    "lightgreen",     # alt-EJ / MMEJ+SSA (end resection)
    # Inner ring -- exactly as in the original static figure
    "inner_nhej_del": "cornflowerblue", # c-NHEJ end trimming (DEL)
    "inner_nhej_ins": "skyblue",        # c-NHEJ fill-in synthesis (INS)
    "inner_alt_del":  "lightcoral",     # alt-EJ / MMEJ+SSA deletions
    "inner_alt_ins":  "gold",           # alt-EJ / MMEJ+SSA insertions
    "edge":           "#FFFFFF",
}

EVENT_COLORS = {
    # INS family: same named-colour family as donut inner ring, all readable on white
    "INS_0_1":     "cornflowerblue",
    "INS_2_4":     "steelblue",
    "INS_5_9":     "skyblue",
    "INS_10_plus": "lightblue",
    # DEL family: same named-colour family as donut inner ring
    "DEL_0_1":     "lightcoral",
    "DEL_2_4":     "indianred",
    "DEL_5_9":     "salmon",
    "DEL_10_plus": "peachpuff",
}
EVENT_LABELS = {
    "INS_0_1":     "Ins 0\u20131 bp",
    "INS_2_4":     "Ins 2\u20134 bp",
    "INS_5_9":     "Ins 5\u20139 bp",
    "INS_10_plus": "Ins \u226510 bp",
    "DEL_0_1":     "Del 0\u20131 bp",
    "DEL_2_4":     "Del 2\u20134 bp",
    "DEL_5_9":     "Del 5\u20139 bp",
    "DEL_10_plus": "Del \u226510 bp",
}

FLAG_COLOR = {
    "robust":         C["robust"],
    "intermediate":   C["intermediate"],
    "sensitive":      C["sensitive"],
    "low_confidence": C["low_conf"],
}


def _despine(ax: plt.Axes, left: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)


def _row_stripe(ax: plt.Axes, n: int, height: float = 1.0) -> None:
    for i in range(0, n, 2):
        ax.axhspan(i - height / 2, i + height / 2,
                   color=C["bg_panel"], zorder=0, linewidth=0)


# ============================================================
# PLOTS
# ============================================================

def plot_pathway_donuts(
    results_df: pd.DataFrame,
    output_file: str,
    variant_name: str = "full",
    mode_name: str = "moderate",
    max_cols: int = 6,
) -> None:
    """
    Nested donut schematic driven by dynamic model output.
    Matches the design language of the original static figure.

    Outer ring: c-NHEJ (blue) vs alt-EJ/MMEJ+SSA (red).
    Inner ring: four sub-fractions --
        c-NHEJ DEL (end trimming, dark blue)
        c-NHEJ INS (fill-in synthesis, pale blue)
        alt-EJ DEL (MMEJ/SSA deletions, dark red)
        alt-EJ INS (alt-EJ insertions, pale red)


    Sorted by descending c-NHEJ fraction.
    Centre text: point estimate. (*) = low confidence.
    """
    sub = results_df[
        (results_df["mode"] == mode_name) & (results_df["variant"] == variant_name)
    ].copy().sort_values("cNHEJ_fraction", ascending=False).reset_index(drop=True)
    if sub.empty:
        return

    n_clades = len(sub)
    ncols = min(max_cols, n_clades)
    nrows = int(np.ceil(n_clades / ncols))

    # Generous per-panel size so labels breathe
    panel_w = 3.0
    panel_h = 3.6
    fig_w = ncols * panel_w
    fig_h = nrows * panel_h + 1.4   # legend row

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(C["bg"])
    axes = np.array(axes).reshape(nrows, ncols)

    outer_r    = 1.00
    ring_w     = 0.30
    inner_r    = outer_r - ring_w   # = 0.70

    for idx in range(nrows * ncols):
        ri = idx // ncols
        ci = idx % ncols
        ax = axes[ri, ci]
        ax.set_aspect("equal")
        ax.set_facecolor(C["bg"])
        ax.axis("off")

        if idx >= n_clades:
            continue

        row = sub.iloc[idx]
        c_frac = row["cNHEJ_fraction"]
        a_frac = row["altEJ_fraction"]

        if pd.isna(c_frac) or pd.isna(a_frac):
            ax.text(0, 0, "no data", ha="center", va="center",
                    fontsize=8, color=C["subtext"])
            continue

        # -- outer ring ------------------------------------------
        outer_sizes  = [c_frac, a_frac]
        outer_colors = [DONUT["outer_nhej"], DONUT["outer_altej"]]
        wedges_outer, _ = ax.pie(
            outer_sizes, startangle=90,
            radius=outer_r,
            wedgeprops=dict(width=ring_w, edgecolor=DONUT["edge"],
                            linewidth=1.2),
            colors=outer_colors,
        )

        # -- inner ring ------------------------------------------
        c_del_f = row.get("cNHEJ_DEL_fraction", np.nan)
        c_ins_f = row.get("cNHEJ_INS_fraction", np.nan)
        a_del_f = row.get("altEJ_DEL_fraction", np.nan)
        a_ins_f = row.get("altEJ_INS_fraction", np.nan)

        c_del_f = 0.5  if pd.isna(c_del_f) else c_del_f
        c_ins_f = 1.0 - c_del_f if pd.isna(c_ins_f) else c_ins_f
        a_del_f = 0.5  if pd.isna(a_del_f) else a_del_f
        a_ins_f = 1.0 - a_del_f if pd.isna(a_ins_f) else a_ins_f

        inner_sizes = [
            c_del_f * c_frac,
            c_ins_f * c_frac,
            a_del_f * a_frac,
            a_ins_f * a_frac,
        ]
        inner_colors = [
            DONUT["inner_nhej_del"],
            DONUT["inner_nhej_ins"],
            DONUT["inner_alt_del"],
            DONUT["inner_alt_ins"],
        ]
        ax.pie(
            inner_sizes, startangle=90,
            radius=inner_r,
            wedgeprops=dict(width=ring_w, edgecolor=DONUT["edge"],
                            linewidth=0.8),
            colors=inner_colors,
        )

        # -- centre annotation ------------------------------------
        low_conf = bool(row.get("low_confidence", False))
        centre_col = C["low_conf"] if low_conf else C["text"]
        ax.text(0, 0.06, f"{c_frac:.2f}",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=centre_col,
                )
        ax.text(0, -0.16, "c-NHEJ",
                ha="center", va="center",
                fontsize=6.5, color=C["subtext"])

        # -- panel title ------------------------------------------
        dagger = "\u2020" if low_conf else ""
        # Place title above the donut at y ~ outer_r + 0.12
        ax.text(0, outer_r + 0.14, f"{row['clade']}{dagger}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=centre_col,
                
                wrap=False)

    # -- shared legend --------------------------------------------
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=DONUT["outer_nhej"],     edgecolor="white",
              label="c-NHEJ  (outer ring)"),
        Patch(facecolor=DONUT["outer_altej"],    edgecolor="white",
              label="alt-EJ / MMEJ+SSA  (outer ring)"),
        Patch(facecolor=DONUT["inner_nhej_del"], edgecolor="white",
              label="c-NHEJ: end trimming (DEL)"),
        Patch(facecolor=DONUT["inner_nhej_ins"], edgecolor="white",
              label="c-NHEJ: fill-in synthesis (INS)"),
        Patch(facecolor=DONUT["inner_alt_del"],  edgecolor="white",
              label="alt-EJ / MMEJ+SSA: deletions"),
        Patch(facecolor=DONUT["inner_alt_ins"],  edgecolor="white",
              label="alt-EJ / MMEJ+SSA: insertions"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.005),
        frameon=True,
        edgecolor=C["rule"],
        handlelength=1.6,
    )
    fig.suptitle(
        f"Dynamic DSBR pathway model \u2014 {mode_name} weighting, {variant_name} variant\n"
        f"Centre value = estimated c-NHEJ fraction \u2020 = n\u202f<\u202f{MIN_RELIABLE_EVENTS} events",
        fontsize=10, y=1.002, color=C["text"],
        
    )

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close()
    log.info("Saved %s", output_file)


def plot_slope_graph(
    results_df: pd.DataFrame,
    output_file: str,
    variant_name: str = "full",
) -> None:
    """
    Slopegraph (parallel coordinates) of c-NHEJ fraction across the three
    weighting modes per clade.

    Each line = one clade. Slope magnitude = sensitivity to prior assumptions.
    Flat lines indicate robust classification; steep lines indicate
    weight-dependent classification.

    Line colour = robustness flag (green / amber / red / grey).
    Clade name at left; moderate point estimate at right.
    Vertical axis lines mark the three mode positions.
    A background white stroke prevents overlap artefacts on dense plots.
    """
    modes = ["conservative", "moderate", "aggressive"]
    sub = results_df[results_df["variant"] == variant_name].copy()
    if sub.empty:
        return

    pivot = sub.pivot(index="clade", columns="mode",
                      values="cNHEJ_fraction").reset_index()
    pivot = pivot.dropna(subset=modes)
    pivot["_range"] = pivot[modes].max(axis=1) - pivot[modes].min(axis=1)
    pivot["_flag"]  = pivot["_range"].apply(
        lambda r: "robust" if r <= ROBUST_THRESHOLD
        else ("sensitive" if r > SENSITIVE_THRESHOLD else "intermediate")
    )
    pivot = pivot.sort_values("moderate", ascending=False).reset_index(drop=True)
    n = len(pivot)

    fig_h = max(5.5, n * 0.40)
    fig, ax = plt.subplots(figsize=(5.2, fig_h))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    # Vertical axis rules
    for xi in [0, 1, 2]:
        ax.axvline(xi, color=C["rule"], linewidth=0.8, zorder=0)

    x_pos = {m: i for i, m in enumerate(modes)}

    for _, row in pivot.iterrows():
        color = FLAG_COLOR.get(row["_flag"], C["subtext"])
        ys = [row[m] for m in modes]
        xs = [x_pos[m] for m in modes]

        # White backing stroke for legibility
        ax.plot(xs, ys, color="white", linewidth=3.5, zorder=1,
                solid_capstyle="round")
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.90, zorder=2,
                solid_capstyle="round")
        ax.scatter(xs, ys, color=color, s=30, zorder=3,
                   edgecolors="white", linewidths=0.8)

        ax.text(-0.07, ys[0], row["clade"], ha="right", va="center",
                fontsize=7.5, color=color)
        ax.text(2.07, ys[-1], f"{row['moderate']:.2f}",
                ha="left", va="center",
                fontsize=7.5, color=color)

    # 0.5 reference line
    ax.axhline(0.5, color=C["grid"], linewidth=0.8,
               linestyle=":", zorder=0, alpha=0.8)
    ax.text(2.07, 0.5, "0.50", ha="left", va="center",
            fontsize=6.5, color=C["subtext"])

    ax.set_xlim(-0.75, 2.72)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Conservative", "Moderate", "Aggressive"],
                       fontsize=9)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1.0"], fontsize=8)
    ax.set_ylabel("Estimated c-NHEJ fraction", fontsize=9)
    ax.yaxis.grid(True, color=C["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(C["rule"])
    ax.spines["bottom"].set_color(C["rule"])
    _despine(ax)

    ax.set_title(
        f"Prior-weight sensitivity  \u2014  {variant_name} variant\n"
        "Slope magnitude = sensitivity to prior assumptions",
        fontsize=9, loc="left", pad=10, color=C["text"],
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=FLAG_COLOR["robust"],
               lw=2.0, label=f"Robust  (\u0394\u2009\u2264\u2009{ROBUST_THRESHOLD})"),
        Line2D([0], [0], color=FLAG_COLOR["intermediate"],
               lw=2.0, label="Intermediate"),
        Line2D([0], [0], color=FLAG_COLOR["sensitive"],
               lw=2.0, label=f"Sensitive  (\u0394\u2009>\u2009{SENSITIVE_THRESHOLD})"),
        Line2D([0], [0], color=FLAG_COLOR["low_confidence"],
               lw=2.0, label=f"Low confidence  (n\u202f<\u202f{MIN_RELIABLE_EVENTS})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7.5,
              handlelength=1.5, borderpad=0.8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close()
    log.info("Saved %s", output_file)


def plot_uncertainty_decomposition(
    combined_df: pd.DataFrame,
    output_file: str,
    variant_name: str = "full",
    mode_name: str = "moderate",
) -> None:
    """
    Horizontal lollipop chart with nested CI capsules, one row per clade.
    Sorted by ascending c-NHEJ fraction (highest clade at top).

    For each clade:
      Outer capsule (wide, pale)  : weight-perturbation 95% CI
      Inner capsule (narrow, mid) : multinomial bootstrap 95% CI
      Filled circle               : point estimate
      Right annotation            : combined SD (+/-)

    Separating the two CI sources is the scientific contribution of this
    figure: a narrow multinomial CI with a wide weight-prior CI signals
    data-consistent but prior-dependent classification.
    """
    sub = combined_df[
        (combined_df["mode"] == mode_name) & (combined_df["variant"] == variant_name)
    ].copy().sort_values("cNHEJ_fraction", ascending=True).reset_index(drop=True)
    if sub.empty:
        return

    n = len(sub)
    row_h = 0.46
    fig_h = max(4.0, n * row_h + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    _row_stripe(ax, n, height=row_h * 2)

    y = np.arange(n, dtype=float)

    for i, row in sub.iterrows():
        low_conf  = bool(row.get("low_confidence", False))
        col_dark  = C["low_conf"]    if low_conf else C["nhej_mid"]
        col_light = C["low_conf"]    if low_conf else C["nhej_pale"]
        col_mid   = C["low_conf"]    if low_conf else C["nhej_light"]
        yi = float(i)
        x_pt = row["cNHEJ_fraction"]

        # -- outer capsule: weight-prior CI ----------------------
        w_lo = row.get("weight_boot_cNHEJ_ci_low",  np.nan)
        w_hi = row.get("weight_boot_cNHEJ_ci_high", np.nan)
        if not (pd.isna(w_lo) or pd.isna(w_hi)):
            ax.barh(yi, w_hi - w_lo, left=w_lo, height=0.40,
                    color=col_light, alpha=0.70,
                    linewidth=0, zorder=1)

        # -- inner capsule: multinomial CI -----------------------
        m_lo = row.get("cNHEJ_fraction_ci_low",  np.nan)
        m_hi = row.get("cNHEJ_fraction_ci_high", np.nan)
        if not (pd.isna(m_lo) or pd.isna(m_hi)):
            ax.barh(yi, m_hi - m_lo, left=m_lo, height=0.22,
                    color=col_mid, alpha=0.90,
                    linewidth=0, zorder=2)

        # -- stem line from 0 to point ---------------------------
        ax.plot([0, x_pt], [yi, yi], color=col_dark,
                linewidth=0.5, alpha=0.30, zorder=1)

        # -- point estimate ---------------------------------------
        edge = "white" if not low_conf else col_dark
        ax.scatter(x_pt, yi, s=52, color=col_dark, zorder=4,
                   edgecolors=edge, linewidths=0.9)

        # -- combined SD annotation -------------------------------
        csd = row.get("cNHEJ_fraction_combined_sd", np.nan)
        if not pd.isna(csd):
            ax.text(1.025, yi, f"\u00b1{csd:.3f}",
                    va="center", ha="left",
                    fontsize=6.5, color=C["subtext"],
                    style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(sub["clade"].tolist(), fontsize=8.5)
    ax.set_xlim(0, 1.14)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Estimated c-NHEJ fraction", fontsize=9)

    # 0.5 reference
    ax.axvline(0.5, color=C["altej_mid"], linewidth=0.7,
               linestyle="--", zorder=0, alpha=0.45)

    ax.xaxis.grid(True, color=C["grid"], linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(C["rule"])
    ax.spines["bottom"].set_color(C["rule"])
    _despine(ax)

    ax.set_title(
        f"Uncertainty decomposition  \u2014  {mode_name} / {variant_name}\n"
        "Pale capsule\u2009=\u2009weight-prior 95% CI     "
        "Dark capsule\u2009=\u2009multinomial 95% CI     "
        "\u00b1\u2009=\u2009combined SD",
        fontsize=8.5, loc="left", pad=10, color=C["text"],
    )

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg = [
        Patch(facecolor=C["nhej_pale"], alpha=0.70,
              label="Weight-prior 95% CI"),
        Patch(facecolor=C["nhej_light"], alpha=0.90,
              label="Multinomial 95% CI"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=C["nhej_mid"], markersize=7,
               markeredgecolor="white", label="Point estimate"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=7.5,
              handlelength=1.4, borderpad=0.8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close()
    log.info("Saved %s", output_file)


def plot_event_composition(
    results_df: pd.DataFrame,
    output_file: str,
    variant_name: str = "full",
    mode_name: str = "moderate",
) -> None:
    """
    Stacked horizontal bar of normalised event-class proportions, one row
    per clade, sorted by ascending c-NHEJ fraction (highest at top).

    Layout:
      Left panel  : stacked bar showing proportions of each size class.
                    INS classes (blue family) always left of DEL classes
                    (red family); a vertical rule and axis-level zone
                    labels mark the boundary.
      Right panel : dumbbell showing c-NHEJ (circle) and alt-EJ (diamond)
                    point estimates on a 0-1 axis.
      Bottom      : shared legend placed below both panels, outside the
                    plot area, so it never overlaps bars.
    """
    sub = results_df[
        (results_df["mode"] == mode_name) & (results_df["variant"] == variant_name)
    ].copy().sort_values("cNHEJ_fraction", ascending=True).reset_index(drop=True)
    if sub.empty:
        return

    n         = len(sub)
    selected  = [e for e in EVENT_ORDER if f"raw_fraction_{e}" in sub.columns]
    frac_cols = [f"raw_fraction_{e}" for e in selected]

    fracs = sub[frac_cols].fillna(0).values
    row_totals = fracs.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    fracs = fracs / row_totals

    # -- figure geometry -------------------------------------------
    bar_h   = 0.62          # height of each stacked bar
    row_h   = 0.52          # vertical space per clade row
    legend_h_in = 0.55      # reserved inches at bottom for legend
    top_pad_in  = 0.70      # title + zone-label row
    data_h_in   = max(3.0, n * row_h)
    fig_h   = data_h_in + legend_h_in + top_pad_in

    fig = plt.figure(figsize=(12.0, fig_h), facecolor=C["bg"])

    # GridSpec: two columns (bars | dots), legend row below
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[4, 1],
        height_ratios=[data_h_in, legend_h_in],
        hspace=0.0,
        wspace=0.04,
    )
    ax_comp = fig.add_subplot(gs[0, 0])
    ax_pts  = fig.add_subplot(gs[0, 1])
    ax_leg  = fig.add_subplot(gs[1, :])   # full-width legend axis

    for ax in (ax_comp, ax_pts, ax_leg):
        ax.set_facecolor(C["bg"])

    # -- row stripes -----------------------------------------------
    for ax in (ax_comp, ax_pts):
        for i in range(0, n, 2):
            ax.axhspan(i - 0.5, i + 0.5,
                       color=C["bg_panel"], zorder=0, linewidth=0)

    y     = np.arange(n, dtype=float)
    lefts = np.zeros(n)
    ins_end = np.zeros(n)

    # -- stacked bars ----------------------------------------------
    for j, e in enumerate(selected):
        vals = fracs[:, j]
        ax_comp.barh(y, vals, left=lefts, height=bar_h,
                     color=EVENT_COLORS[e], linewidth=0, zorder=2)
        if e in INS_EVENTS:
            ins_end = lefts + vals
        lefts += vals

    # -- INS / DEL boundary ----------------------------------------
    ins_boundary = float(np.median(ins_end[ins_end > 0])) \
                   if (ins_end > 0).any() else 0.5
    ax_comp.axvline(ins_boundary, color=C["rule"], linewidth=1.0,
                    linestyle="-", zorder=3, alpha=0.7)

    # Zone labels placed in the top margin via annotate with
    # axes-fraction y so they sit above the data area, not inside it
    ax_comp.annotate(
        "<- Insertions",
        xy=(ins_boundary * 0.5, 1.0), xycoords=("data", "axes fraction"),
        xytext=(0, 6), textcoords="offset points",
        ha="center", va="bottom", fontsize=8,
        color=C["nhej_mid"],
        annotation_clip=False,
    )
    ax_comp.annotate(
        "Deletions ->",
        xy=(ins_boundary + (1 - ins_boundary) * 0.5, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 6), textcoords="offset points",
        ha="center", va="bottom", fontsize=8,
        color=C["altej_mid"],
        annotation_clip=False,
    )

    # -- left panel axes -------------------------------------------
    ax_comp.set_yticks(y)
    ax_comp.set_yticklabels(sub["clade"].tolist(), fontsize=8.5)
    ax_comp.set_xlim(0, 1)
    ax_comp.set_ylim(-0.6, n - 0.4)
    ax_comp.set_xlabel("Proportion of events (normalised within clade)",
                       fontsize=9)
    ax_comp.xaxis.grid(True, color=C["grid"], linewidth=0.4, zorder=0)
    ax_comp.set_axisbelow(True)
    ax_comp.spines["left"].set_color(C["rule"])
    ax_comp.spines["bottom"].set_color(C["rule"])
    _despine(ax_comp)
    ax_comp.set_title(
        f"Event-class composition  \u2014  {mode_name} / {variant_name}",
        fontsize=10, loc="left", pad=18, color=C["text"],
    )

    # -- right panel: dumbbell -------------------------------------
    for i, row in sub.iterrows():
        ax_pts.plot(
            [row["cNHEJ_fraction"], row["altEJ_fraction"]],
            [i, i],
            color=C["rule"], linewidth=1.0, zorder=1,
        )
    ax_pts.scatter(sub["cNHEJ_fraction"].values, y,
                   color=C["nhej_mid"], s=50, zorder=3,
                   edgecolors="white", linewidths=0.9)
    ax_pts.scatter(sub["altEJ_fraction"].values, y,
                   color=C["altej_mid"], s=50, zorder=3,
                   edgecolors="white", linewidths=0.9,
                   marker="D")
    ax_pts.axvline(0.5, color=C["grid"], linewidth=0.7,
                   linestyle="--", zorder=0)
    ax_pts.set_xlim(0, 1)
    ax_pts.set_ylim(-0.6, n - 0.4)
    ax_pts.set_yticks(y)
    ax_pts.set_yticklabels([])
    ax_pts.set_xlabel("Pathway\nfraction", fontsize=8)
    ax_pts.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax_pts.xaxis.grid(True, color=C["grid"], linewidth=0.4, zorder=0)
    ax_pts.set_axisbelow(True)
    ax_pts.spines["left"].set_color(C["rule"])
    ax_pts.spines["bottom"].set_color(C["rule"])
    _despine(ax_pts)
    ax_pts.set_title("c-NHEJ\nvs alt-EJ",
                     fontsize=8, pad=18, color=C["text"])

    # -- legend axis: entirely outside data area -------------------
    ax_leg.axis("off")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    ins_handles = [Patch(facecolor=EVENT_COLORS[e], linewidth=0,
                         label=EVENT_LABELS[e])
                   for e in selected if e in INS_EVENTS]
    del_handles = [Patch(facecolor=EVENT_COLORS[e], linewidth=0,
                         label=EVENT_LABELS[e])
                   for e in selected if e in DEL_EVENTS]
    pt_handles  = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=C["nhej_mid"], markersize=7,
               markeredgecolor="white", label="c-NHEJ fraction"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=C["altej_mid"], markersize=6,
               markeredgecolor="white", label="alt-EJ / MMEJ+SSA fraction"),
    ]
    all_handles = ins_handles + del_handles + pt_handles
    ax_leg.legend(
        handles=all_handles,
        loc="center",
        ncol=5,
        fontsize=8,
        frameon=True,
        edgecolor=C["rule"],
        handlelength=1.4,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close()
    log.info("Saved %s", output_file)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    log.info("DSBR pathway model v2 -- starting run")
    write_provenance()
    validate_weights(WEIGHT_MODES)

    df = pd.read_csv(INPUT_FILE)
    log.info("Input: %d rows, %d columns", *df.shape)
    df = validate_input(df)

    df.to_csv(f"{OUTPUT_DIR}/validated_input.tsv", sep="\t", index=False)
    log.info("Validated input saved.")

    all_results, all_boot, all_wboot = [], [], []

    for variant_name, selected_events in MODEL_VARIANTS.items():
        for mode_name, weights in WEIGHT_MODES.items():
            log.info("Processing: variant=%s  mode=%s", variant_name, mode_name)

            res = compute_scores_for_mode(
                df=df, mode_name=mode_name, weights=weights,
                variant_name=variant_name, selected_events=selected_events,
            )
            boot = bootstrap_multinomial_all(
                df=df, mode_name=mode_name, weights=weights,
                variant_name=variant_name, selected_events=selected_events,
                n_boot=N_BOOT, seed=RANDOM_SEED,
            )
            wboot = weight_perturbation_boot(
                df=df, mode_name=mode_name, weights=weights,
                variant_name=variant_name, selected_events=selected_events,
                n_boot=N_WEIGHT_BOOT, seed=RANDOM_SEED + 99999,
            )
            all_results.append(res)
            all_boot.append(boot)
            all_wboot.append(wboot)

    results_df = pd.concat(all_results, ignore_index=True)
    boot_df    = pd.concat(all_boot,    ignore_index=True)
    wboot_df   = pd.concat(all_wboot,   ignore_index=True)

    merged_df   = results_df.merge(boot_df, on=["clade", "mode", "variant"], how="left")
    combined_df = combine_uncertainties(merged_df, wboot_df)

    # Save core outputs
    results_df.to_csv(f"{OUTPUT_DIR}/dynamic_model_results.tsv",        sep="\t", index=False)
    boot_df.to_csv(f"{OUTPUT_DIR}/dynamic_model_bootstrap.tsv",         sep="\t", index=False)
    wboot_df.to_csv(f"{OUTPUT_DIR}/dynamic_model_weight_bootstrap.tsv", sep="\t", index=False)
    combined_df.to_csv(f"{OUTPUT_DIR}/dynamic_model_combined.tsv",      sep="\t", index=False)

    # Per-variant stability
    for variant_name in MODEL_VARIANTS:
        pivot_df, rank_df, corr_df = compare_rank_stability(results_df, variant_name)
        pivot_df.to_csv(f"{OUTPUT_DIR}/mode_comparison_{variant_name}.tsv",  sep="\t", index=False)
        rank_df.to_csv(f"{OUTPUT_DIR}/rank_stability_{variant_name}.tsv",    sep="\t", index=False)
        corr_df.to_csv(f"{OUTPUT_DIR}/rank_correlation_{variant_name}.tsv",  sep="\t", index=False)
        log.info(
            "Rank correlations (%s): %s", variant_name,
            corr_df[["mode_A", "mode_B", "kendall_tau", "spearman_rho"]].to_dict("records"),
        )
        sens_df = mode_sensitivity_table(results_df, variant_name)
        sens_df.to_csv(f"{OUTPUT_DIR}/mode_sensitivity_{variant_name}.tsv", sep="\t", index=False)

    # Variant sensitivity per mode
    for mode_name in WEIGHT_MODES:
        var_cmp = compare_variants_within_mode(results_df, mode_name)
        var_cmp.to_csv(f"{OUTPUT_DIR}/variant_sensitivity_{mode_name}.tsv", sep="\t", index=False)

    # Pairwise bootstrap significance tests
    # Primary analysis: moderate / full
    # Also run across all mode x variant combinations for sensitivity
    pairwise_rows = []
    for v_name in MODEL_VARIANTS:
        for m_name in WEIGHT_MODES:
            pw = pairwise_bootstrap_tests(
                df, mode_name=m_name, variant_name=v_name,
                n_boot=N_BOOT, seed=RANDOM_SEED,
            )
            pairwise_rows.append(pw)
    pairwise_df = pd.concat(pairwise_rows, ignore_index=True)
    pairwise_df.to_csv(
        f"{OUTPUT_DIR}/pairwise_significance.tsv", sep="\t", index=False
    )
    log.info("Pairwise significance tests saved.")

    # Driver summary
    driver_df = identify_top_drivers(results_df)
    driver_df.to_csv(f"{OUTPUT_DIR}/top_drivers_summary.tsv", sep="\t", index=False)

    # Run summary
    summary_df = make_run_summary(results_df, combined_df)
    summary_df.to_csv(f"{OUTPUT_DIR}/run_summary_moderate_full.tsv", sep="\t", index=False)

    # Plots:
    #   1. pathway_donuts   : schematic model result figure (equivalent to original static figure)
    #   2. slope_graph      : robustness across weighting modes (slopegraph)
    #   3. uncertainty_decomp: dual CI decomposition per clade
    #   4. event_composition: raw event proportions with pathway fractions
    for variant_name in MODEL_VARIANTS:
        plot_pathway_donuts(
            results_df,
            f"{OUTPUT_DIR}/pathway_donuts_{variant_name}.png",
            variant_name=variant_name,
            mode_name="moderate",
        )
        plot_slope_graph(
            results_df,
            f"{OUTPUT_DIR}/slope_mode_sensitivity_{variant_name}.png",
            variant_name=variant_name,
        )
        plot_uncertainty_decomposition(
            combined_df,
            f"{OUTPUT_DIR}/uncertainty_decomp_{variant_name}.png",
            variant_name=variant_name,
            mode_name="moderate",
        )
        plot_event_composition(
            results_df,
            f"{OUTPUT_DIR}/event_composition_{variant_name}.png",
            variant_name=variant_name,
            mode_name="moderate",
        )

    log.info("Run complete. Outputs in: %s", OUTPUT_DIR)
    log.info("")
    log.info("Key outputs:")
    log.info("  %s/run_summary_moderate_full.tsv      <- start here", OUTPUT_DIR)
    log.info("  %s/pathway_donuts_full.png            <- model schematic (replaces static figure)", OUTPUT_DIR)
    log.info("  %s/slope_mode_sensitivity_full.png    <- weight-mode robustness slopegraph", OUTPUT_DIR)
    log.info("  %s/uncertainty_decomp_full.png        <- CI source decomposition", OUTPUT_DIR)
    log.info("  %s/event_composition_full.png         <- data-to-model traceability", OUTPUT_DIR)
    log.info("  %s/rank_correlation_full.tsv          <- rank stability statistics", OUTPUT_DIR)
    log.info("  %s/pairwise_significance.tsv          <- bootstrap pairwise tests (all modes x variants)", OUTPUT_DIR)
    log.info("  %s/provenance.json                    <- full run metadata", OUTPUT_DIR)


if __name__ == "__main__":
    main()
