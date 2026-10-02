#!/usr/bin/env python3
"""
pools.py — customer-pool generation strategies for the arm comparison.

Every strategy returns a sorted list of customer ids drawn from
`instance.C`. The depot is never included; `alns.ALNS` adds it.

Arms
----
B   random_pool             drawn uniformly at random
C1  greedy_prize            highest prize
C2  greedy_penalty          highest penalty (most expensive to skip)
C3  greedy_tw_spread        time windows tiling the planning horizon
C4  greedy_cluster          chosen cluster by cluster, keeping clusters whole
C5  greedy_value_density    best (prize + penalty) per unit detour
D   (screened elsewhere)    best of ~500 candidates, ranked by the GAT

Every arm selects the same number of customers, so all face the same
structural handicap and differences reflect strategy alone. Every deterministic
arm pool is also placed in D's candidate set, so the surrogate chooses from a
superset of what the baselines would pick and can only lose by misranking. surrogate

Randomized-greedy candidates
----------------------------
`randomized_greedy` follows the usual restricted-candidate-list scheme: at each
of the 85 steps it ranks the remaining customers by the criterion and draws
uniformly from the top `k`. k = 1 reproduces the deterministic greedy pool and
large k approaches uniform sampling; the default k = 8 gives diverse pools that
still respect the criterion.
"""

import random
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


DEFAULT_POOL_SIZE = 85
DEFAULT_RCL_SIZE = 8
DEFAULT_N_CLUSTERS = 10
CRITERIA = ("prize", "penalty", "tw_spread", "cluster", "density")


# ---------------------------------------------------------------------------
# Deterministic strategies
# ---------------------------------------------------------------------------
def random_pool(instance, size: int = DEFAULT_POOL_SIZE,
                rng: Optional[random.Random] = None) -> List[int]:
    """
    Arm B: uniform random sample — the null strategy and the usual baseline.

    Matches `generate_non_repeating_integers(1, n, 85)` in main_ALNS.ipynb,
    so this arm reproduces the sampling that generated the training data.
    """
    rng = rng or random
    return sorted(rng.sample(list(instance.C), size))


def greedy_prize(instance, size: int = DEFAULT_POOL_SIZE) -> List[int]:
    """C1: the `size` customers with the highest prize."""
    ranked = sorted(instance.C, key=lambda c: (-instance.reward_val[c], c))
    return sorted(ranked[:size])


def greedy_penalty(instance, size: int = DEFAULT_POOL_SIZE) -> List[int]:
    """
    C2: the `size` customers with the highest penalty.

    These are the most expensive to leave unserved, so pooling them gives the
    search the best chance of avoiding the largest penalty terms.
    """
    ranked = sorted(instance.C, key=lambda c: (-instance.pen_val[c], c))
    return sorted(ranked[:size])


def _window_midpoints(instance) -> Dict[int, float]:
    """Midpoint of each customer's [release_time, deadline] window."""
    return {c: 0.5 * (instance.time_windows[c][0] + instance.time_windows[c][1])
            for c in instance.C}


def _horizon(instance) -> tuple:
    """
    Planning horizon spanned by the customer windows.

    Depot deadlines are excluded: the depot commonly carries a sentinel
    deadline (1e9 in these instances) that would otherwise stretch the horizon
    and collapse every slice into the first bucket.
    """
    releases = [instance.time_windows[c][0] for c in instance.C]
    deadlines = [instance.time_windows[c][1] for c in instance.C]
    return min(releases), max(deadlines)


def greedy_tw_spread(instance, size: int = DEFAULT_POOL_SIZE) -> List[int]:
    """
    C3: a pool whose time windows tile the horizon evenly.

    Cuts the horizon into `size` equal slices and, for each slice in turn,
    takes the unused customer whose window midpoint is nearest that slice's
    centre. When a slice has no unused customer nearby the nearest remaining
    one is taken anyway, so exactly `size` customers are always returned.

    The intent is a pool the vehicle can schedule smoothly, rather than one
    whose demands all fall in the same part of the day.
    """
    lo, hi = _horizon(instance)
    mid = _window_midpoints(instance)
    centres = np.linspace(lo, hi, size, endpoint=True) if size > 1 else [0.5 * (lo + hi)]

    remaining = set(instance.C)
    chosen: List[int] = []
    for centre in centres:
        if not remaining:
            break
        pick = min(remaining, key=lambda c: (abs(mid[c] - centre), c))
        chosen.append(pick)
        remaining.discard(pick)
    return sorted(chosen)


def _kmeans(points: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    """
    Minimal deterministic k-means with k-means++ seeding.

    Written out rather than pulled from scikit-learn to avoid adding a
    dependency for twenty lines, and because a fixed seed here keeps pool
    generation reproducible across runs.

    Returns a label per point.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    k = max(1, min(k, n))

    centres = [points[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(((points[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        centres.append(points[rng.choice(n, p=d2 / total) if total > 0 else rng.integers(n)])
    centres = np.array(centres, dtype=float)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        new_labels = dist.argmin(axis=1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for c in range(k):
            members = points[labels == c]
            if len(members):
                centres[c] = members.mean(axis=0)
    return labels


def _cluster_members(instance, n_clusters: int, seed: int = 0) -> Dict[int, List[int]]:
    """Group customers into spatial clusters, keyed by cluster id."""
    ids = list(instance.C)
    pts = np.array([instance.locations[c] for c in ids], dtype=float)
    labels = _kmeans(pts, n_clusters, seed=seed)
    out: Dict[int, List[int]] = {}
    for cid, lab in zip(ids, labels):
        out.setdefault(int(lab), []).append(cid)
    return out


def greedy_cluster(instance, size: int = DEFAULT_POOL_SIZE,
                   n_clusters: int = DEFAULT_N_CLUSTERS, seed: int = 0) -> List[int]:
    """
    C4: fill the pool cluster by cluster, keeping clusters intact.

    Clusters are ranked by mean value density — (prize + penalty) divided by
    distance from the depot — and taken whole until the next one would not fit;
    the remainder is filled from that cluster's own best customers.

    The motivation is that random sampling cuts across clusters, leaving the
    vehicle to travel out to a cluster region and serve only part of it. Keeping
    clusters whole preserves the geometry that makes clustered instances
    tractable, which matters most for C and RC topologies.
    """
    clusters = _cluster_members(instance, n_clusters, seed=seed)
    tt = instance.traveling_time

    def value(c: int) -> float:
        return (instance.reward_val[c] + instance.pen_val[c]) / max(tt[(0, c)], 1e-9)

    ranked = sorted(clusters.items(),
                    key=lambda kv: (-float(np.mean([value(c) for c in kv[1]])), kv[0]))

    chosen: List[int] = []
    for _, members in ranked:
        if len(chosen) + len(members) <= size:
            chosen.extend(members)
        else:
            room = size - len(chosen)
            if room > 0:
                chosen.extend(sorted(members, key=lambda c: (-value(c), c))[:room])
            break
    # A fragmented clustering can leave the pool short; top it up by value.
    if len(chosen) < size:
        rest = sorted(set(instance.C) - set(chosen), key=lambda c: (-value(c), c))
        chosen.extend(rest[:size - len(chosen)])
    return sorted(chosen[:size])


def _density_scores(instance) -> Dict[int, float]:
    """(prize + penalty) per unit distance from the depot."""
    tt = instance.traveling_time
    return {c: (instance.reward_val[c] + instance.pen_val[c]) / max(tt[(0, c)], 1e-9)
            for c in instance.C}


def greedy_value_density(instance, size: int = DEFAULT_POOL_SIZE) -> List[int]:
    """
    C5: the `size` customers with the best value per unit of travel.

    Combines both objective terms a customer contributes — its prize if served
    and its penalty if skipped — against a crude proxy for what it costs to
    reach. Unlike C1 and C2, which each optimise one term in isolation, this
    trades value against distance.
    """
    scores = _density_scores(instance)
    return sorted(sorted(instance.C, key=lambda c: (-scores[c], c))[:size])


# ---------------------------------------------------------------------------
# Randomized greedy (candidate generation for arm D)
# ---------------------------------------------------------------------------
def randomized_greedy(
    instance,
    criterion: str,
    size: int = DEFAULT_POOL_SIZE,
    k: int = DEFAULT_RCL_SIZE,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """
    A randomized variant of one greedy criterion.

    At each step the remaining customers are ranked by `criterion` and one is
    drawn uniformly from the top `k` (the restricted candidate list). For
    "tw_spread" the ranking is by distance to the current slice centre, so the
    tiling structure is preserved while the exact choice varies.
    """
    rng = rng or random
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")

    remaining = set(instance.C)
    chosen: List[int] = []

    if criterion == "cluster":
        # Walk clusters in a randomised order weighted toward the denser ones,
        # taking each whole. Varies which clusters make the cut rather than
        # which customers within them, so pools stay geometrically coherent.
        clusters = _cluster_members(instance, DEFAULT_N_CLUSTERS,
                                    seed=rng.randrange(10_000))
        scores = _density_scores(instance)
        order = sorted(clusters.items(),
                       key=lambda kv: (-float(np.mean([scores[c] for c in kv[1]])), kv[0]))
        pool_of_clusters = [m for _, m in order]
        while pool_of_clusters and len(chosen) < size:
            pick = pool_of_clusters.pop(rng.randrange(min(3, len(pool_of_clusters))))
            room = size - len(chosen)
            if len(pick) <= room:
                chosen.extend(pick)
            else:
                chosen.extend(sorted(pick, key=lambda c: (-scores[c], c))[:room])
        if len(chosen) < size:
            rest = sorted(set(instance.C) - set(chosen), key=lambda c: (-scores[c], c))
            chosen.extend(rest[:size - len(chosen)])
        return sorted(chosen[:size])

    if criterion in ("prize", "penalty", "density"):
        values = ({"prize": instance.reward_val,
                   "penalty": instance.pen_val}.get(criterion)
                  or _density_scores(instance))
        for _ in range(min(size, len(remaining))):
            rcl = sorted(remaining, key=lambda c: (-values[c], c))[:k]
            pick = rng.choice(rcl)
            chosen.append(pick)
            remaining.discard(pick)
    else:  # tw_spread
        lo, hi = _horizon(instance)
        mid = _window_midpoints(instance)
        centres = np.linspace(lo, hi, size, endpoint=True) if size > 1 else [0.5 * (lo + hi)]
        for centre in centres:
            if not remaining:
                break
            rcl = sorted(remaining, key=lambda c: (abs(mid[c] - centre), c))[:k]
            pick = rng.choice(rcl)
            chosen.append(pick)
            remaining.discard(pick)

    return sorted(chosen)


# ---------------------------------------------------------------------------
# Candidate set for arm D
# ---------------------------------------------------------------------------
def build_candidate_set(
    instance,
    n_candidates: int = 500,
    size: int = DEFAULT_POOL_SIZE,
    k: int = DEFAULT_RCL_SIZE,
    rng: Optional[random.Random] = None,
    include_pools: Optional[Dict[str, List[int]]] = None,
) -> tuple:
    """
    Build the candidate set the surrogate ranks for arm D.

    The pools used by arms B, C1, C2 and C3 are inserted first (via
    `include_pools`), so the candidate set is a superset of what every other
    arm would pick. Arm D can therefore only lose to another arm when the
    surrogate misranks — the comparison measures screening quality rather than
    two different sampling procedures.

    The remainder is split evenly across the three randomized-greedy criteria.
    Duplicates are dropped, so slightly fewer than `n_candidates` pools may be
    returned; the count is reported alongside.

    Returns (candidates, origins) — parallel lists of pools and of the label
    describing where each came from.
    """
    rng = rng or random
    candidates: List[List[int]] = []
    origins: List[str] = []
    seen = set()

    def add(pool: Sequence[int], origin: str) -> None:
        key = tuple(sorted(pool))
        if key not in seen:
            seen.add(key)
            candidates.append(list(key))
            origins.append(origin)

    for label, pool in (include_pools or {}).items():
        add(pool, label)

    remaining = max(0, n_candidates - len(candidates))

    # Three sources, so the surrogate is not confined to pools the baseline
    # heuristics would themselves produce:
    #   randomized greedy  -- near-greedy variants across every criterion
    #   perturbed          -- neighbourhoods of the deterministic arm pools
    #   uniform random     -- unstructured, for coverage
    n_rgreedy = int(round(0.60 * remaining))
    n_perturb = int(round(0.25 * remaining))
    n_random = remaining - n_rgreedy - n_perturb

    per_criterion = n_rgreedy // len(CRITERIA)
    leftover = n_rgreedy - per_criterion * len(CRITERIA)
    for i, criterion in enumerate(CRITERIA):
        for _ in range(per_criterion + (1 if i < leftover else 0)):
            add(randomized_greedy(instance, criterion, size=size, k=k, rng=rng),
                f"rgreedy_{criterion}")

    seeds = [(lab, pool) for lab, pool in (include_pools or {}).items()
             if len(pool) == size]
    if seeds and n_perturb > 0:
        outside_all = set(instance.C)
        for i in range(n_perturb):
            lab, base_pool = seeds[i % len(seeds)]
            keep = list(base_pool)
            outside = sorted(outside_all - set(keep))
            if not outside:
                continue
            # Swap out a random 10-20% of the pool for customers outside it.
            # Capped by how many are actually outside: with 85 of 100 pooled
            # only 15 are available, so an uncapped 20% swap would drop more
            # than it could add and return a short pool.
            n_swap = max(1, min(int(round(rng.uniform(0.10, 0.20) * size)),
                                len(keep), len(outside)))
            drop = rng.sample(keep, n_swap)
            gain = rng.sample(outside, n_swap)
            add(sorted(set(keep) - set(drop) | set(gain)), f"perturb_{lab}")

    for _ in range(max(0, n_random)):
        add(random_pool(instance, size, rng), "random")

    return candidates, origins
