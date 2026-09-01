"""
Correlating weight-only measures against ground-truth generalization.

Rank statistics rather than Pearson: the relationship between a measure and
test accuracy need not be linear, and rank correlation is what the
generalization-measure literature reports. Implemented here because the
project has no scipy.
"""

import numpy as np


def rank(a):
    """Ranks, averaging ties -- the transform that turns Pearson into Spearman."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)

    # average ranks within tied groups
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.arange(i, j + 1).mean()
        i = j + 1
    return ranks


def kendall_tau(x, y):
    """
    Kendall's tau-b: (concordant - discordant) normalized with a tie correction.

    O(n^2), which is irrelevant at 25 models.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    concordant = discordant = 0
    tied_x = tied_y = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if dx == 0:
                    tied_x += 1
                if dy == 0:
                    tied_y += 1

    n0 = n * (n - 1) / 2
    denom = np.sqrt(max(n0 - tied_x, 0) * max(n0 - tied_y, 0))
    return float((concordant - discordant) / denom) if denom > 0 else 0.0


def partial_spearman(x, y, control):
    """
    Rank correlation between x and y with `control` regressed out.

    A measure that correlates with test accuracy only because both track weight
    scale will lose that correlation here. This is the check that separates
    "measures structure" from "measures magnitude".
    """
    rx, ry, rz = rank(x), rank(y), rank(control)

    def pearson(a, b):
        a, b = a - a.mean(), b - b.mean()
        denom = np.sqrt((a**2).sum() * (b**2).sum())
        return float((a * b).sum() / denom) if denom > 0 else 0.0

    r_xy, r_xz, r_yz = pearson(rx, ry), pearson(rx, rz), pearson(ry, rz)
    denom = np.sqrt(max(1 - r_xz**2, 0) * max(1 - r_yz**2, 0))
    return float((r_xy - r_xz * r_yz) / denom) if denom > 1e-12 else 0.0


def _check():
    """Known-answer checks for both statistics."""
    assert abs(kendall_tau([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 1.0) < 1e-12
    assert abs(kendall_tau([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1.0) < 1e-12

    # hand-computed: pairs (1,2) and (1,3) concordant, (2,3) discordant -> (2-1)/3
    assert abs(kendall_tau([1, 2, 3], [1, 3, 2]) - 1 / 3) < 1e-12

    # the canonical partial-correlation case: x and y share a driver z but have
    # no direct relationship, so the raw correlation is high and the partial ~0
    rng = np.random.default_rng(0)
    z = rng.standard_normal(400)
    x = z + 0.3 * rng.standard_normal(400)
    y = z + 0.3 * rng.standard_normal(400)
    raw = abs(partial_spearman(x, y, rng.standard_normal(400)))
    controlled = abs(partial_spearman(x, y, z))
    assert raw > 0.8, f"raw correlation should be high, got {raw:.3f}"
    assert controlled < 0.2, f"controlled correlation should collapse, got {controlled:.3f}"

    print(f"generalization stats checks passed "
          f"(shared-driver correlation {raw:.2f} -> {controlled:.2f} after control)")
    return True


if __name__ == "__main__":
    _check()
