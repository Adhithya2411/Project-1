"""Significance testing and effect sizes for system comparisons.

improvement.txt §5 asks for paired bootstrap tests, 95% confidence intervals,
and Cohen's *d* on every comparison. This module exists so those are computed
one way rather than five, because the experiment scripts had each grown their
own version and one of them was wrong in an instructive way.

A note on the p-value, since this is the part that is easy to get subtly wrong.
The natural-looking construction is:

    resample the paired differences; report the fraction of resamples
    where the difference is <= 0

That quantity is not a p-value. It estimates ``P(delta <= 0 | data)`` — a
posterior-style credibility statement — whereas a p-value is
``P(observing something at least this extreme | H0 true)``. The two coincide
only under particular priors, and the first is systematically more likely to
declare significance. The bootstrap hypothesis test instead re-centres the
difference distribution so the null is true by construction, then asks how
often a resample from *that* distribution is as extreme as what was observed.
:func:`paired_bootstrap` does the latter.
"""

from __future__ import annotations

import numpy as np

#: Fixed so every reported interval is reproducible (improvement.txt §9a).
DEFAULT_SEED = 1729
DEFAULT_RESAMPLES = 10_000


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for a mean.

    Args:
        values: Per-item metric values.
        confidence: Interval width, e.g. 0.95.
        resamples: Bootstrap resample count.
        seed: RNG seed.

    Returns:
        ``(mean, lower, upper)``. All zeros for an empty input, so a caller can
        print a row without special-casing.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0, 0.0, 0.0
    if values.size == 1:
        single = float(values[0])
        return single, single, single

    rng = np.random.RandomState(seed)
    # Vectorised: one (resamples x n) index matrix beats a Python loop by ~50x,
    # which matters because every metric of every system calls this.
    indices = rng.randint(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(values.mean()),
        float(np.percentile(means, alpha * 100.0)),
        float(np.percentile(means, (1.0 - alpha) * 100.0)),
    )


def paired_bootstrap(
    treatment: np.ndarray,
    control: np.ndarray,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Two-sided paired bootstrap test on ``treatment - control``.

    The differences are re-centred to zero mean, making the null hypothesis
    ("no difference") true in the resampling distribution. The p-value is the
    fraction of resampled means at least as far from zero as the observed mean.

    Args:
        treatment: Per-item values for the system under test.
        control: Per-item values for the reference system, item-aligned.
        resamples: Bootstrap resample count.
        seed: RNG seed.

    Returns:
        ``(observed_mean_difference, p_value)``. Returns ``(0.0, 1.0)`` when
        the inputs are empty or misaligned, and ``(delta, 1.0)`` when every
        paired difference is identical — there is no variance to test against,
        so no evidence against the null.

    Raises:
        ValueError: If the two arrays have different lengths. Silently
            truncating would pair unrelated items, which is the failure mode
            this signature exists to prevent.
    """
    treatment = np.asarray(treatment, dtype=np.float64).ravel()
    control = np.asarray(control, dtype=np.float64).ravel()
    if treatment.size != control.size:
        raise ValueError(
            f"paired_bootstrap requires item-aligned arrays; got "
            f"{treatment.size} and {control.size}. Filter both systems with "
            f"the same mask before comparing."
        )
    if treatment.size == 0:
        return 0.0, 1.0

    differences = treatment - control
    observed = float(differences.mean())
    spread = float(differences.std(ddof=1)) if differences.size > 1 else 0.0
    if spread <= 1e-15:
        return observed, 1.0

    rng = np.random.RandomState(seed)
    centred = differences - differences.mean()
    indices = rng.randint(0, centred.size, size=(resamples, centred.size))
    null_means = centred[indices].mean(axis=1)
    p_value = float((np.abs(null_means) >= abs(observed)).mean())
    return observed, p_value


def cohens_d(treatment: np.ndarray, control: np.ndarray) -> float:
    """Cohen's *d* for paired samples: mean difference over its own SD.

    The paired form is the right one here because both systems answer the same
    queries; using the pooled independent-sample SD would understate the effect
    by ignoring that pairing.

    Returns:
        0.0 when there is no variation in the differences, which is the
        degenerate case where an effect size is not defined.
    """
    treatment = np.asarray(treatment, dtype=np.float64).ravel()
    control = np.asarray(control, dtype=np.float64).ravel()
    if treatment.size != control.size or treatment.size < 2:
        return 0.0
    differences = treatment - control
    spread = float(differences.std(ddof=1))
    if spread <= 1e-15:
        return 0.0
    return float(differences.mean() / spread)


def interpret_d(value: float) -> str:
    """Label an effect size using Cohen's conventional thresholds.

    Conventions, not laws — reported alongside the number so a reader can
    disagree with the label without having to recompute anything.
    """
    magnitude = abs(value)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def significance_marker(p_value: float) -> str:
    """Conventional star notation, or ``ns`` when not significant."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def summarise_comparison(
    treatment: np.ndarray,
    control: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Full comparison record: means, CIs, delta, p-value, and effect size."""
    treatment_mean, treatment_low, treatment_high = bootstrap_ci(treatment, confidence)
    control_mean, control_low, control_high = bootstrap_ci(control, confidence)
    delta, p_value = paired_bootstrap(treatment, control)
    effect = cohens_d(treatment, control)
    return {
        "n": int(np.asarray(treatment).size),
        "treatment_mean": treatment_mean,
        "treatment_ci95": [treatment_low, treatment_high],
        "control_mean": control_mean,
        "control_ci95": [control_low, control_high],
        "delta": delta,
        "p_value": p_value,
        "significant_at_05": bool(p_value < 0.05),
        "cohens_d": effect,
        "effect_size": interpret_d(effect),
        "marker": significance_marker(p_value),
    }


__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "bootstrap_ci",
    "cohens_d",
    "interpret_d",
    "paired_bootstrap",
    "significance_marker",
    "summarise_comparison",
]
