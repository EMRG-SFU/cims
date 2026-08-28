"""
Market-share calibration: fit fixed intangible costs, and optionally technology
lifetimes, so that modelled market shares reproduce a calibration counterfactual.

Two entry points:

    optimize_ms_via_fics_v2(model, node, **kwargs)
        Fit one FIC per technology per year.

    optimize_ms_via_fics_and_lifetimes(model, node, **kwargs)
        The same fit, plus a search for shorter lifetimes on technologies whose
        modelled share does not fall as fast as the counterfactual asks.

Both mutate the model they are given and return a dict of diagnostics.


How the fit is posed
--------------------

**The competition, inverted.** At a tech-compete node a technology's share of new
stock is proportional to `softplus(lcc_competition) ** -v`, where `v` is the
node's heterogeneity. Only the RATIO of lifecycle costs between two technologies
sets the ratio of their shares, so each year is seeded in closed form: one
technology is taken as the reference — the one with the largest counterfactual
share that year, the best-conditioned denominator — and every other technology's
required cost follows from

    lcc_i = lcc_ref * (target_i / target_ref) ** (-1 / v)

The gap between that required cost and the technology's actual cost becomes a FIC
by dividing through `d(lcc_competition)/d(fic)`, measured numerically once per
year. This seeding is the single most important part of the method. Once a
technology's share reaches 0 or 1 the logit is exponentially flat in its FIC and
the gradient vanishes, so a gradient method started from a flat point returns
that point unchanged. Starting from the analytic solution puts the optimizer
somewhere the gradient is informative. Continuing each year from the previous
year's solution (`start='warm'`) walks straight into the flat region and freezes
the FICs at identical values in every year.

**Scaled variables.** A FIC runs to hundreds or thousands while the objective is
a sum of share differences of order one, so raw gradient components come out at
1e-3 to 1e-7 and a solver's default step and finite-difference sizes are
meaningless against them. The fitted variable is `z = fic / fic_scale` with
`fic_scale = 100`, which puts the free variables at order one. Every penalty and
bound below is expressed in those same scaled units, so the weights mean the same
thing at any node.

**A smooth objective.** The quantity minimised is the sum of SQUARED share
differences. L-BFGS-B assumes a smooth objective, and an absolute difference is
kinked exactly at the solution, which stalls the line search. The reported
diagnostic is still the L1 sum, so the numbers stay readable as "total share
error".

**Explicit tolerances.** `ftol` and `gtol` are set separately and directly.
Passing scipy's single `tol` to L-BFGS-B sets BOTH, and a `gtol` loose enough to
be a sensible function tolerance is satisfied at the starting point by gradients
this small — the optimizer then returns without moving.

**Only responsive technologies are fitted.** Removed from the free variables:
the base year (all market shares are exogenous there), technologies outside their
available/unavailable window, and any technology whose objective response to a
scale-aware two-sided FIC probe is numerically zero.


The answer is not unique, and what to do about it
-------------------------------------------------

Wherever the competition saturates — one technology taking essentially all of the
new share — a wide range of FIC vectors produce exactly the same shares. Adding a
constant across a wide window can leave the objective unchanged to seven
significant figures. Nothing in the share objective picks a point on that
plateau, and an unpenalised fit lands somewhere arbitrary on it, with a series
that can swing by a hundred or more between adjacent years.

`ridge` adds `ridge * sum(z ** 2)`: a preference for the smallest FICs that fit,
which pins down the arbitrary LEVEL.

`smooth` adds `smooth * sum((z_year - z_neighbour) ** 2)`: a preference against
year-to-year movement, which pins down the arbitrary PATH. `smooth_passes`
sweeps the year range more than once, so each year is anchored to both its
neighbours rather than only the one before it.

**`ridge` is not a portable constant — scale it to the node.** Its marginal cost
grows linearly in the FIC while the share error it can buy back is bounded by the
squared residual, so it caps the fitted FICs at whatever magnitude the crossover
happens to fall at. Where a node's counterfactual asks a technology to fall to a
fraction of a percent, suppressing it takes a very large cost penalty, and a
ridge weight that was almost free elsewhere becomes the binding constraint.
Compare two nodes, same weight:

    node                                   ridge     fit L1    max |FIC|
    14 techs, mild counterfactual            0        3.401       145
                                           1e-4       3.047        83
    13 techs, one tech pushed to 0.003       0        1.706      4139
                                           1e-5       2.568       702
                                           1e-4       5.113       467

At the first node `ridge=1e-4` improves the fit AND halves the magnitudes. At the
second it triples the error, because the fit needs FICs near 2000 and the penalty
overtakes the objective in the low hundreds. Budget one fit at `ridge=0` to see
which regime a node is in. Where the aim is a steady series rather than small
numbers, prefer `smooth`, which penalises movement and is indifferent to level.

`pin_reference` is a third option — hold one technology at `fic = 0` so the rest
read as premiums against it — and it measures badly wherever it has been tried,
inflating the largest FICs several-fold. The intuition behind it, that only
relative lifecycle costs matter so the level is arbitrary, does not hold: the
competition is a power law in `softplus(lcc)`, not an exponential in `lcc`, so a
constant added to every cost compresses the ratios and does change shares. It is
kept for experimentation.


Zero-target technologies
------------------------

A logit hands every competing technology a strictly positive share, so a
counterfactual share of exactly zero is only reachable as the FIC goes to
infinity — and the objective keeps improving, by less and less, the further out
the solver walks. Carrying those technologies as free variables sends the
optimizer chasing that limit, produces implausible FICs, and is where most of the
run time goes at a node with many of them.

`zero_targets='freeze'` (the default) instead suppresses each one analytically,
aiming the same closed-form inversion at `suppress_floor` rather than at zero,
and then holds it fixed. The technology is pushed to a negligible share, the FIC
that does it stays a sane number, and the solver spends its evaluations on
technologies whose targets it can actually hit. The same floor guards the seed,
so a target of zero never becomes a division by zero.

`target_floor` is a blunter alternative: clip the targets themselves away from
exactly 0. Prefer marking such technologies unavailable in the model description
where that is what is meant.


Run time
--------

`skip_retrofits`: `calc_retrofits` is roughly 40% of an objective evaluation. 
It is gated by `retrofit_existing_max`, and where that is 0 for every 
technology in every year the competition runs and returns nothing, so
skipping it changes no result and saves 35-45%. `'auto'`, the default, checks
that and skips only where it is safe. Note this changes what is being modelled
during the search rather than how the solver searches, which is why it is
guarded.


What a FIC cannot fix
---------------------

The objective targets `market_share_total`, which includes surviving vintage
stock, while a FIC only moves `market_share_new`. Once a technology's new share
has been driven near zero, what remains is its own past purchases decaying at its
lifetime, and no FIC reaches them. A technology sitting persistently above a
falling counterfactual, with a large FIC already applied, is usually showing
stock rather than a failure of the fit.

That is the gap `optimize_ms_via_fics_and_lifetimes` exists to close: it searches
for a shorter lifetime, which is the retirement lever. Note that lifetime has two
effects, both pushing share down — it sets the retirement rate, and where
`capital_recovery` is not defined at the technology it is also the payback period
in the capital recovery factor, so a shorter lifetime raises annualised capital
cost. The FIC fit absorbs part of that second effect, so a fitted lifetime is not
a pure retirement statement.

Two things no fit can fix at all, and worth checking when a residual will not
close: a counterfactual that assigns share to a technology the model has already
retired through its availability window, and a counterfactual that asks a
technology to grow while the technologies it would grow out of are unavailable.
Both are model-description problems, and the roster printed by the lifetime
search names the technologies where they show up.


Using it
--------

    from Calibration.Optimization.optimize_ms_v2 import (
        optimize_ms_via_fics_v2, optimize_ms_via_fics_and_lifetimes)

    # FICs only, smallest that fit
    res = optimize_ms_via_fics_v2(model, node, ridge=1e-5)
    total_error = sum(r['end'] for r in res.values())

    # FICs and lifetimes, with a figure at every step
    res = optimize_ms_via_fics_and_lifetimes(model, node, ridge=1e-5, plot=True)
    res['lifetimes']       # tech -> fitted lifetime
    res['changed']         # only the technologies that were shortened
    res['rosters']         # what was considered at each pass, and why not

Both functions mutate `model` in place, so re-load it between runs that are meant
to be compared. Per-year solver output goes to `logFile`, not to the console;
`verbose` controls the progress summary.
"""

import os

import numpy as np
import scipy.optimize as SO
from contextlib import contextmanager, redirect_stdout, redirect_stderr

from CIMS.utils.parameter import list as PARAM

import Calibration.Data.node_info as node_info
from Calibration.Optimization._objectiveFunctions import make_objective_localNode
from Calibration.CIMS_Functions.set_param_calibration import set_param_calibration

DEAD_TECH_PROBE = 10.0      # FIC perturbation used to detect non-responding techs
DEAD_TECH_TOL = 1e-12       # objective change below this counts as no response
SLOPE_PROBE = 100.0         # FIC perturbation used to measure d(lcc)/d(fic)
SEED_FLOOR = 1e-4           # smallest target share the analytic seed will aim at
SEED_CLIP = 5000.0          # cap on the magnitude of a seeded FIC


def _no_retrofits(model, node, year, stock_existing):
    """Stand-in for `calc_retrofits`: no retrofit competition happens."""
    return stock_existing, {}, {}


def _retrofits_are_active(model, node, all_techs, all_years):
    """
    Whether retrofits can actually do anything at this node.

    `calc_retrofits` is gated by `retrofit_existing_max` — the share of a
    technology's existing stock that may be retrofitted away. Where that is 0 for
    every technology in every year, the retrofit competition runs and returns
    nothing, so skipping it changes no result.
    """
    for tech in all_techs:
        for year in all_years:
            try:
                limit = model.get_param(PARAM.retrofit_existing_max, node, year, tech=tech)
            except Exception:
                return True                   # cannot tell — assume they matter
            if limit:
                return True
    return False


@contextmanager
def _retrofits_disabled(enabled):
    """
    Temporarily replace `calc_retrofits` inside the stock allocation module.

    `stock_allocation.py` does `from .retrofits import calc_retrofits`, so the
    name has to be patched in that module's namespace rather than in
    `retrofits`. Restored on exit, including on error.
    """
    if not enabled:
        yield
        return

    from CIMS.stock_allocation import stock_allocation as _sa
    original = _sa.calc_retrofits
    _sa.calc_retrofits = _no_retrofits
    try:
        yield
    finally:
        _sa.calc_retrofits = original


def _competing_techs(model, node, year, all_techs):
    """Technologies inside their available/unavailable window in `year`."""
    base = str(model.base_year)
    live = []
    for tech in all_techs:
        try:
            avail = model.get_param(PARAM.available, node, base, tech=tech)
            unavail = model.get_param(PARAM.unavailable, node, base, tech=tech)
        except Exception:
            live.append(tech)
            continue
        if avail is None or unavail is None or avail <= int(year) < unavail:
            live.append(tech)
    return live


def _probe_magnitude(model, node, year, all_techs, live, apply):
    """
    FIC perturbation large enough to reorder the competition.

    Market share depends on the SPREAD of competition lifecycle costs, so a
    probe has to be able to cross that spread to tell whether a technology can
    respond at all. Scaled by the measured `d(lcc)/d(fic)` and floored at
    `DEAD_TECH_PROBE`.
    """
    costs = []
    for tech in live:
        value = model.get_param(PARAM.lcc_competition, node, year, tech=tech)
        if value is not None:
            costs.append(float(value))
    if len(costs) < 2:
        return DEAD_TECH_PROBE

    spread = max(costs) - min(costs)
    idx = all_techs.index(live[0]) if live else 0
    slope = _lcc_slope(model, node, year, all_techs, idx, apply)
    if not slope:
        slope = 1.0
    return max(DEAD_TECH_PROBE, 2.0 * spread / abs(slope))


def _lcc_slope(model, node, year, all_techs, probe_idx, apply):
    """
    Measure d(lcc_competition)/d(fic) numerically for one technology.

    FIC enters competition annual cost as `(fom + fic + dic) / output`, so the
    slope is 1/output — constant across technologies at a node, but measured
    rather than assumed. Returns None if the response is not usable.
    """
    tech = all_techs[probe_idx]
    before = model.get_param(PARAM.lcc_competition, node, year, tech=tech)
    probe = np.zeros(len(all_techs))
    probe[probe_idx] = SLOPE_PROBE
    apply(probe)
    after = model.get_param(PARAM.lcc_competition, node, year, tech=tech)
    apply(np.zeros(len(all_techs)))
    if before is None or after is None:
        return None
    slope = (after - before) / SLOPE_PROBE
    return slope if abs(slope) > 1e-9 else None


def _pin_index(model, node, all_techs, pin_reference):
    """
    Index of the technology to hold at fic = 0, or None.

    'base_year_share' picks the technology with the largest `market_share_total`
    in the base year. Market shares depend only on RELATIVE lifecycle costs, so
    without a pinned reference the FIC level is arbitrary; fixing the dominant
    technology at zero makes every other FIC read as a premium or discount
    against it.
    """
    if not pin_reference:
        return None
    if pin_reference in all_techs:
        return all_techs.index(pin_reference)
    if pin_reference != 'base_year_share':
        raise ValueError(f"pin_reference not recognised: {pin_reference!r}")

    base = str(model.base_year)
    best, best_share = None, None
    for i, tech in enumerate(all_techs):
        try:
            share = model.get_param(PARAM.market_share_total, node, base, tech=tech)
        except Exception:
            share = None
        if share is None:
            continue
        if best_share is None or share > best_share:
            best, best_share = i, share
    return best


def _analytic_seed(model, node, year, all_techs, targets, free_idx, apply, ref=None):
    """
    Invert the market-share logit to get a starting FIC vector.

    For a tech-compete node the new market share of technology i is
    proportional to `softplus(lcc_competition_i) ** -v`, so matching a target
    share relative to the reference technology requires

        lcc_i = lcc_ref * (target_i / target_ref) ** (-1 / v)

    Falls back to zeros if heterogeneity, the lifecycle costs, or the FIC slope
    cannot be read.
    """
    n_all = len(all_techs)
    seed = np.zeros(n_all)
    if not free_idx:
        return seed

    try:
        v = float(model.get_param(PARAM.heterogeneity, node, year))
    except (TypeError, ValueError):
        return seed
    if not v:
        return seed

    slope = _lcc_slope(model, node, year, all_techs, free_idx[0], apply)
    if slope is None:
        return seed

    lcc = {}
    for i in range(n_all):
        lcc[i] = model.get_param(PARAM.lcc_competition, node, year, tech=all_techs[i])

    if ref is None:
        ref = int(np.argmax(targets))
    if lcc.get(ref) is None or targets[ref] <= 0:
        return seed

    for i in free_idx:
        if i == ref or lcc.get(i) is None:
            continue
        wanted = lcc[ref] * (max(targets[i], SEED_FLOOR) / targets[ref]) ** (-1.0 / v)
        seed[i] = float(np.clip((wanted - lcc[i]) / slope, -SEED_CLIP, SEED_CLIP))

    return seed


def optimize_ms_via_fics_v2(
        model,
        nodeName,
        fic_scale=100.0,
        fic_min=None,
        start='analytic',
        smooth=0.0,
        smooth_passes=1,
        restarts=0,
        target_floor=None,
        zero_targets='freeze',
        suppress_floor=1e-4,
        skip_retrofits='auto',
        ridge=0.0,
        pin_reference=None,
        ftol=1e-10,
        gtol=1e-7,
        maxiter=5000,
        maxfun=20000,
        skip_base_year=True,
        seed=0,
        logFile="log_optimize_ms_via_fics.log",
        verbose=True):
    """
    Calibrate FICs so modelled market shares match the calibration counterfactual.

    Parameters
    ----------
    start : {'analytic', 'warm', 'zero', 'analytic+warm'}
        Where each year's optimization starts. 'analytic' inverts the market
        share logit for that year's costs and targets — recommended, and the
        only option that gives FICs which actually vary year to year. 'warm'
        continues from the previous year's solution, which freezes the FICs once
        shares reach 0 or 1. 'analytic+warm' tries both and keeps the better
        current-year fit; note that a better fit in one year can leave worse
        inherited stock for the next, so it is not always better overall.
    smooth : float
        Weight on a penalty against year-to-year FIC movement, added to the
        squared share error as `smooth * sum((z_year - z_neighbour) ** 2)` with
        `z = fic / fic_scale`. 0 disables it. Because the share objective is
        flat over wide regions of FIC space, this buys a far steadier FIC series
        for very little fit — see the table in the module docstring.
    smooth_passes : int
        How many times to sweep the whole year range. The first sweep can only
        anchor a year to the year before it; later sweeps anchor to both
        neighbours, which is what keeps the series from drifting forward.
        Pointless when `smooth` is 0.
    zero_targets : {'freeze', 'free', 'drop'}
        How to treat technologies whose counterfactual share is zero. A logit
        gives every competing technology a strictly positive share, so a zero
        target is only approachable as FIC goes to infinity — carrying those
        technologies as free variables makes the optimizer chase that limit and
        is where most of the run time goes at a node with many of them.
        'freeze' (default) suppresses them once, analytically, aiming at
        `suppress_floor`, then holds them fixed. 'free' optimizes them like any
        other technology. 'drop' leaves their FIC at 0.
    suppress_floor : float
        Share the analytic suppression aims a zero-target technology at.
    skip_retrofits : {'auto', True, False}
        Replace `calc_retrofits` with a no-op for the duration of the fit.
        Retrofits are about 40% of an objective evaluation, so this is the
        largest single saving available in the inner loop — but it changes the
        model rather than the solver, and FICs fitted without retrofits would
        not reproduce a run that has them.

        'auto' (the default) skips them only where they cannot do anything:
        `calc_retrofits` is gated by `retrofit_existing_max`, and where that is 0
        for every technology in every year the competition runs and returns
        nothing, so the fit is bit-identical and 35-45% faster. Where the limit is
        non-zero, 'auto' keeps retrofits and says so, because skipping would then
        change the answer rather than just the run time. `True` forces the skip
        regardless; `False` always keeps them.
    ridge : float
        Weight on `ridge * sum(z ** 2)`, z = fic / fic_scale — a preference for
        the smallest FICs that fit. The share objective says nothing about FIC
        magnitude and is flat over wide regions, so without this (or
        `pin_reference`) the level is arbitrary. NOT a portable constant: it caps
        the fitted FICs at the magnitude where its marginal cost overtakes the
        share error, which at a node needing large FICs can dominate the fit
        entirely. See the module docstring for measured values and how to check
        which regime a node is in.
    pin_reference : str or None
        Technology held at fic = 0, either a technology name or
        `'base_year_share'` to pick the one with the largest base-year
        `market_share_total`. Market shares depend only on relative lifecycle
        costs, so pinning removes the arbitrary level and makes every other FIC
        read as a premium or discount against the dominant technology. Free.

    Returns
    -------
    dict : keyed by year, each entry holding
        'start'   – L1 market-share error before optimization
        'end'     – L1 market-share error after optimization
        'fics'    – {tech: fic} applied to the model
        'free'    – technologies that were optimized in that year
        'result'  – the raw scipy OptimizeResult
    """
    all_techs = node_info.list_techs(model.graph, nodeName)
    all_years = node_info.list_years(model.graph, nodeName)
    rng = np.random.default_rng(seed)
    n_all = len(all_techs)

    warm = {tech: 0.0 for tech in all_techs}
    out = {}
    responsive_cache = {}
    pin_idx = _pin_index(model, nodeName, all_techs, pin_reference)

    fit_years = [y for y in all_years
                 if not (skip_base_year and int(y) == model.base_year)]
    # FIC path being solved for, {year: full-length FIC vector}. Used as the
    # anchor for the temporal smoothness penalty.
    path = {y: np.zeros(n_all) for y in fit_years}
    solved = set()

    # Each sweep re-solves every year; with smoothing on, later sweeps can also
    # anchor a year to the year that follows it, not just the one before.
    schedule = [(sweep, year)
                for sweep in range(max(1, int(smooth_passes)))
                for year in fit_years]

    if skip_retrofits == 'auto':
        active = _retrofits_are_active(model, nodeName, all_techs, all_years)
        skipping = not active
        if active and verbose:
            print("retrofits kept: retrofit_existing_max is non-zero at this node, "
                  "so skipping them would change the result")
    else:
        skipping = bool(skip_retrofits)

    with open(logFile, 'w') as fh, redirect_stdout(fh), redirect_stderr(fh), \
            _retrofits_disabled(skipping):
        for sweep, year in schedule:

            objective = make_objective_localNode(model, nodeName, year, all_techs)

            def apply(vec):
                """Full-length FIC vector -> objective detail dict."""
                return objective(vec, retAll=True)

            zeros = np.zeros(n_all)
            start_detail = apply(zeros)
            targets = list(start_detail['y'])
            if target_floor is not None:
                targets = [max(t, target_floor) for t in targets]

            def l1(vec):
                d = apply(vec)
                return sum(abs(a - b) for a, b in zip(targets, d['y_est']))

            start_l1 = l1(zeros)

            # ---- choose the free variables ------------------------------------
            # Which technologies can move the objective at all. Cached: this costs
            # one evaluation per technology and does not change between sweeps.
            if year in responsive_cache:
                responsive = responsive_cache[year]
            else:
                live = _competing_techs(model, nodeName, year, all_techs)
                # The probe has to be big enough to move the logit and has to be
                # tried in BOTH directions. A technology sitting at 0% of new
                # share does not react to being made more expensive, and one
                # sitting at 100% with a wide cost margin does not react to a
                # small penalty — a one-sided nudge classifies both as dead, and
                # a technology classified dead keeps fic = 0, which for the
                # saturated one means it keeps winning everything.
                magnitude = _probe_magnitude(model, nodeName, year, all_techs,
                                             live, apply)
                responsive = []
                for i, tech in enumerate(all_techs):
                    if tech not in live:
                        continue
                    moved = False
                    for signed in (magnitude, -magnitude):
                        probe = np.zeros(n_all)
                        probe[i] = signed
                        if abs(apply(probe)['totalDiff']
                               - start_detail['totalDiff']) > DEAD_TECH_TOL:
                            moved = True
                            break
                    if moved:
                        responsive.append(i)
                apply(zeros)
                responsive_cache[year] = responsive

            # Technologies whose counterfactual share is zero cannot be fitted —
            # a logit gives every competing technology a strictly positive share,
            # so the optimizer walks their FIC toward infinity and burns
            # iterations doing it. Suppress them once, analytically, and hold
            # them fixed rather than carrying them as free variables.
            zero_idx = [i for i in responsive if targets[i] <= 0]
            free_idx = [i for i in responsive if targets[i] > 0]
            fixed = np.zeros(n_all)

            # The fallback below keys on `zero_targets` alone, and deliberately
            # does NOT free everything when no responsive technology has a positive
            # target. A year like that is one where the counterfactual says every
            # purchasable technology should have no share; freeing them to avoid an
            # empty optimization lets the optimizer hand one of them 100% of new
            # stock, which then persists for that technology's whole lifetime. Such
            # years are suppressed and skipped instead.
            if zero_targets == 'free':
                free_idx, zero_idx = responsive, []
            if zero_targets == 'freeze' and zero_idx:
                lifted = [max(t, suppress_floor) if i in zero_idx else t
                          for i, t in enumerate(targets)]
                suppression = _analytic_seed(model, nodeName, year, all_techs,
                                             lifted, responsive, apply, ref=pin_idx)
                for i in zero_idx:
                    fixed[i] = suppression[i]
            # zero_targets == 'drop' leaves them at fic = 0

            # The pinned reference technology stays at fic = 0 and is not fitted.
            if pin_idx is not None and pin_idx in free_idx and len(free_idx) > 1:
                free_idx = [i for i in free_idx if i != pin_idx]
                fixed[pin_idx] = 0.0

            if not free_idx:
                end_detail = apply(fixed)
                end_l1 = sum(abs(a - b) for a, b in zip(targets, end_detail['y_est']))
                out[year] = {'start': start_l1, 'end': end_l1,
                             'fics': {all_techs[i]: float(fixed[i]) for i in range(n_all)},
                             'free': [],
                             'targets': {all_techs[i]: float(targets[i])
                                         for i in range(n_all)},
                             'estimates': {all_techs[i]: float(end_detail['y_est'][i])
                                           for i in range(n_all)},
                             'result': None}
                path[year] = fixed
                solved.add(year)
                continue

            # ---- smoothness anchors -------------------------------------------
            # Neighbouring years' FICs, in scaled units. The previous year is
            # available from this sweep; the following year only once a sweep has
            # already solved it.
            y_pos = fit_years.index(year)
            anchors = []
            if smooth:
                for nb in (y_pos - 1, y_pos + 1):
                    if 0 <= nb < len(fit_years) and fit_years[nb] in solved:
                        anchors.append(path[fit_years[nb]] / fic_scale)

            # ---- objective in scaled coordinates ------------------------------
            def sum_sq(z):
                vec = fixed.copy()
                for zi, i in zip(z, free_idx):
                    vec[i] = zi * fic_scale
                d = apply(vec)
                value = float(sum((a - b) ** 2 for a, b in zip(targets, d['y_est'])))
                for anchor in anchors:
                    value += smooth * float(sum((zi - anchor[i]) ** 2
                                                for zi, i in zip(z, free_idx)))
                if ridge:
                    value += ridge * float(np.dot(z, z))
                return value

            lo = -np.inf if fic_min is None else fic_min / fic_scale
            bounds = [(lo, None)] * len(free_idx)
            opts = {'maxiter': maxiter, 'maxfun': maxfun, 'ftol': ftol, 'gtol': gtol}

            warm_start = np.array([warm[all_techs[i]] / fic_scale for i in free_idx])
            starts = []
            if 'analytic' in start:
                seeded = _analytic_seed(model, nodeName, year, all_techs,
                                        targets, free_idx, apply, ref=pin_idx)
                starts.append(np.array([max(seeded[i] / fic_scale, lo) for i in free_idx]))
            if 'warm' in start:
                starts.append(warm_start)
            if 'zero' in start or not starts:
                starts.append(np.zeros(len(free_idx)))
            for _ in range(restarts):
                starts.append(rng.uniform(max(lo, -5.0), 5.0, len(free_idx)))

            best = None
            for z0 in starts:
                res = SO.minimize(sum_sq, z0, method="L-BFGS-B", bounds=bounds, options=opts)
                if best is None or res.fun < best.fun:
                    best = res

            vec = fixed.copy()
            for zi, i in zip(best.x, free_idx):
                vec[i] = zi * fic_scale
            end_l1 = l1(vec)

            # Leave the model holding the optimum, and carry it into the next year.
            end_detail = apply(vec)
            for i, tech in enumerate(all_techs):
                warm[tech] = float(vec[i])
            path[year] = vec
            solved.add(year)

            out[year] = {
                'start': start_l1,
                'end': end_l1,
                'fics': {all_techs[i]: float(vec[i]) for i in range(n_all)},
                'free': [all_techs[i] for i in free_idx],
                'targets': {all_techs[i]: float(targets[i]) for i in range(n_all)},
                'estimates': {all_techs[i]: float(end_detail['y_est'][i])
                              for i in range(n_all)},
                'result': best,
            }

    if verbose:
        print(f"{'year':<6}{'start L1':>10}{'end L1':>10}{'free':>6}")
        for year, r in out.items():
            print(f"{year:<6}{r['start']:>10.4f}{r['end']:>10.4f}{len(r['free']):>6}")
        print(f"{'TOTAL':<6}{sum(r['start'] for r in out.values()):>10.4f}"
              f"{sum(r['end'] for r in out.values()):>10.4f}")

    return out


# ---------------------------------------------------------------------------
# Lifetime as a second fitted output
# ---------------------------------------------------------------------------

LIFETIME_LADDER = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def _set_lifetime(model, node, tech, value, years):
    """Write one lifetime value into every year for `tech`.

    Retirement reads `lifetime` at the VINTAGE year (`_base_stock_retirement` and
    `_purchased_stock_retirement` both look it up at the year the stock was
    bought), so a single per-technology lifetime has to be written to every year
    to behave as one number.
    """
    for year in years:
        set_param_calibration(model, float(value), PARAM.lifetime, node,
                              year=year, tech=tech, save=False)


def _purchasable_years(model, node, years, tech):
    """How many of `years` the technology is inside its availability window."""
    base = str(model.base_year)
    try:
        avail = model.get_param(PARAM.available, node, base, tech=tech)
        unavail = model.get_param(PARAM.unavailable, node, base, tech=tech)
    except Exception:
        return len(years)
    if avail is None or unavail is None:
        return len(years)
    return sum(1 for y in years if avail <= int(y) < unavail)


def _longest_nonrising(values, rise_tol=0.002):
    """
    Longest stretch over which a series never rises, in years.

    Returns (length_in_years, start_index). Flat counts as non-rising, which
    matters: a counterfactual pinned at 0 for the whole period never "declines"
    in a strict sense, yet it is the clearest possible statement that the
    technology should not be gaining share. A strictly-falling test excludes
    exactly the technologies most in need of a shorter lifetime.

    `rise_tol` allows a wiggle of up to half a percentage point year to year, so
    a counterfactual digitised from a chart does not have its run chopped into
    pieces. The stretch as a whole must still end no higher than it started, so
    the tolerance cannot accumulate into a rise.
    """
    if len(values) < 2:
        return len(values), 0
    best_len, best_start = 1, 0
    start = 0
    for i in range(1, len(values) + 1):
        broken = (i == len(values)
                  or values[i] > values[i - 1] + rise_tol
                  or values[i] > values[start] + rise_tol)
        if broken:
            if i - start > best_len:
                best_len, best_start = i - start, start
            start = i
        # a broken run restarts at the year that broke it
    return best_len, best_start


def _lifetime_roster(fit, all_techs, model=None, node=None, lifetimes=None,
                     min_decline_run=5, min_overshoot=0.005, peak_overshoot=0.05,
                     max_techs=None, exclude=()):
    """
    Report EVERY technology in the competition, whether or not it is a candidate
    for a lifetime reduction, with the reason it was or was not selected.

    A technology is a candidate when

      1. its counterfactual share is flat or falling for at least
         `min_decline_run` consecutive years somewhere in the fitted range, and
      2. from the first year of that decline onward, its modelled share sits
         above the counterfactual by a mean of `min_overshoot` or more, or by
         `peak_overshoot` or more in some single year.

    Everything else gets a reason instead.

    Both tests are shaped by what goes wrong with the obvious alternatives.
    Comparing only the first and last year of the counterfactual misses the
    rise-then-fall shape a technology displaced part-way through the period
    actually has, and admits one that merely ends lower than it started with no
    sustained fall anywhere — hence a run test. Flat has to count as non-rising: a
    counterfactual pinned at 0 never declines in a strict sense while being the
    clearest possible statement that the technology should not be holding share,
    and a strictly-falling test excludes exactly the technologies most in need of
    a shorter lifetime. On the second test, counting how many later years are
    above target throws away technologies whose overshoot is large but
    concentrated where it matters, and keeps ones a hair above target for most of
    the period — hence a magnitude test with a peak arm. Measuring the overshoot
    from the start of the non-rising stretch rather than over a fixed tail puts
    the window where the lifetime lever is supposed to be acting.

    Technologies that are outside their availability window for the whole fitted
    range but still hold stock are ordered FIRST, regardless of score. They cannot
    receive new stock, so `_find_competing_weights` drops them from the
    competition and no FIC has any effect on them — retirement is the only lever
    there is. Ranking them against technologies the FIC fit can still move would
    let a bigger but fixable overshoot crowd them out.

    Returns a list of dicts, one per technology, in print order: the ones that
    will be tested (in testing order, `order` 1..n), then the candidates cut by
    `max_techs`, then the ineligible ones worst-overshoot first. Each row carries
    `tech`, `score` (mean overshoot from the start of the decline, counting only
    the years above target), `peak` (largest single-year overshoot in that
    window), `run` (longest flat-or-falling stretch of the counterfactual, in
    years),
    `window` (the years the overshoot was measured over), `stock_only`,
    `purchasable`, `n_years`, `lifetime`, `eligible`, `tested`, `order` and
    `reason`.

    `exclude` names technologies to report but never select — used by the second
    roster check, where a technology that has already walked its ladder must not
    be re-opened against a fit its own reduction produced.
    """
    years = sorted(fit.keys())
    if not years:
        return []
    lifetimes = lifetimes or {}
    rows = []

    for tech in all_techs:
        row = {'tech': tech, 'score': 0.0, 'peak': 0.0, 'run': 0,
               'window': (), 'stock_only': False, 'purchasable': None,
               'n_years': len(years), 'lifetime': lifetimes.get(tech),
               'eligible': False, 'tested': False, 'order': None, 'reason': ''}
        rows.append(row)

        try:
            targets = [fit[y]['targets'][tech] for y in years]
            excess = [fit[y]['estimates'][tech] - fit[y]['targets'][tech]
                      for y in years]
        except KeyError:
            row['reason'] = 'no fitted result for this technology'
            continue

        purchasable = (_purchasable_years(model, node, years, tech)
                       if model is not None and node is not None else len(years))
        holds_stock = any(fit[y]['estimates'][tech] > min_overshoot for y in years)
        row['purchasable'] = purchasable
        row['stock_only'] = purchasable == 0 and holds_stock

        run_len, run_start = _longest_nonrising(targets)
        row['run'] = run_len

        # The overshoot is measured from the year the counterfactual stops
        # rising to the end of the range: that is the stretch a shorter lifetime
        # is meant to bend, and it is where an overshoot is a real miss rather
        # than a technology that has simply not been displaced yet.
        window = excess[run_start:] if run_len else excess
        row['window'] = (years[run_start], years[-1]) if run_len else ()
        row['score'] = float(sum(e for e in window if e > 0) / len(window))
        row['peak'] = float(max(window))

        if tech in exclude:
            row['reason'] = 'ladder already walked — not re-opened'
            continue
        if tech in lifetimes and lifetimes[tech] is None:
            row['reason'] = 'no lifetime defined at the technology'
            continue
        if run_len < min_decline_run:
            row['reason'] = (f'counterfactual is rising — never flat or falling for '
                             f'{min_decline_run} consecutive years '
                             f'(longest stretch {run_len}y)')
            continue
        if row['peak'] <= 0:
            row['reason'] = ('modelled share is already at or below target '
                             'throughout that stretch')
            continue
        if row['score'] < min_overshoot and row['peak'] < peak_overshoot:
            row['reason'] = (f'overshoot too small — mean {row["score"]:.4f} '
                             f'(< {min_overshoot:g}), peak {row["peak"]:.4f} '
                             f'(< {peak_overshoot:g})')
            continue

        row['eligible'] = True

    # Among the candidates, stock-only technologies lead (see above); the
    # ineligible remainder is simply worst overshoot first.
    rows.sort(key=lambda r: (not r['eligible'],
                             not (r['stock_only'] and r['eligible']),
                             -r['score']))

    order = 0
    for row in rows:
        if not row['eligible']:
            continue
        if max_techs is not None and order >= max_techs:
            row['reason'] = f'ranked past max_techs={max_techs}'
            continue
        order += 1
        row['tested'] = True
        row['order'] = order
        row['reason'] = ('stock only: outside its availability window all period, '
                         'so no FIC can move it'
                         if row['stock_only']
                         else f'purchasable in {row["purchasable"]} of {row["n_years"]} years')
    return rows


def _print_roster(rows, node):
    """Print the full competition and the lifetime testing order it implies."""
    name_w = max([len(r['tech']) for r in rows] + [30])
    print(f"\ntechnologies in the competition at {node}  ({len(rows)}):")
    print(f"   {'#':>2}  {'technology':<{name_w}}  {'lifetime':>8}  {'no rise':>7}"
          f"  {'overshoot':>9}  {'peak':>7}  status")
    for row in rows:
        num = str(row['order']) if row['order'] else '-'
        life = f"{row['lifetime']:g}" if row['lifetime'] is not None else '-'
        run = f"{row['run']}y" if row['run'] else '-'
        verb = 'test' if row['tested'] else 'not tested'
        print(f"   {num:>2}  {row['tech']:<{name_w}}  {life:>8}  {run:>7}  "
              f"{row['score']:9.4f}  {row['peak']:7.4f}  {verb} \u2014 {row['reason']}")
    tested = [r for r in rows if r['tested']]
    if tested:
        print("   no rise = longest stretch the counterfactual is flat or falling; "
              "overshoot = mean excess over target from the start of that stretch on")
        print(f"   testing order: {', '.join(r['tech'] for r in tested)}")
    else:
        print("   no technology overshoots a declining counterfactual "
              "\u2014 nothing to do")


def _kept_note(accepted, original):
    """One-line summary of the reductions kept so far, for a figure title."""
    if not accepted:
        return 'none'
    return ', '.join(f"{t} {original[t]:g}->{v:g}" for t, v in accepted.items())


def _plot_step(model, node, label, plot_kwargs, figures, aggregate=True,
               verbose=True):
    """
    Render the node's market-share series for one step of the search.

    Calls `plot_ms_for_node.plot_ms_line`, which builds its own figure and calls
    `fig.show()` on it. To label each step and keep the figures afterwards,
    `plotly.io.show` is temporarily wrapped: the wrapper prefixes the figure
    title with `label`, stores the figure, and then shows it as normal. Each call
    produces a separate figure, so nothing is overwritten — in an interactive
    session they stack up as separate outputs, and the whole list comes back in
    the result as 'figures' so they can be re-rendered or written out later.

    Never raises. A missing plotting dependency or a renderer that cannot display
    (a headless run, say) reports itself once and lets the fit continue.
    """
    try:
        from Calibration.CIMS_Functions.aggregation_traversal import aggregation_traversal
        import Calibration.Plotting.plot_ms_for_node as plotMS
        import plotly.io as pio
    except Exception as exc:                       # plotly/plotting not installed
        if verbose:
            print(f"   [plot] skipped, plotting is unavailable: {exc}")
        return

    try:
        if aggregate:
            # Market shares are already on the graph, but the aggregated
            # quantities the plot may draw on are not until this runs.
            with open(os.devnull, 'w') as fh, redirect_stdout(fh):
                aggregation_traversal(model)
    except Exception as exc:
        if verbose:
            print(f"   [plot] aggregation_traversal failed: {exc}")

    real_show = pio.show

    def labelled_show(fig, *args, **kwargs):
        try:
            base = fig.layout.title.text or ''
        except Exception:
            base = ''
        fig.update_layout(title=f"{label}<br><sub>{base}</sub>" if base else label)
        figures.append(fig)
        return real_show(fig, *args, **kwargs)

    pio.show = labelled_show
    try:
        plotMS.plot_ms_line(model, node, **(plot_kwargs or {}))
    except Exception as exc:
        if verbose:
            print(f"   [plot] could not render '{label}': {exc}")
    finally:
        pio.show = real_show


def optimize_ms_via_fics_and_lifetimes(
        model,
        nodeName,
        lifetime_ladder=LIFETIME_LADDER,
        lifetime_min=3.0,
        min_gain=0.02,
        max_techs=3,
        min_decline_run=5,
        min_overshoot=0.005,
        peak_overshoot=0.05,
        plot=False,
        plot_kwargs=None,
        plot_aggregate=True,
        search_kwargs=None,
        verbose=True,
        **fit_kwargs):
    """
    Fit FICs, and additionally shorten the lifetime of technologies whose
    modelled share does not decline fast enough against the counterfactual.

    Lifetime is a single number per technology, applied to every year, and is
    only ever reduced. The search is deliberately conservative: candidates are
    tried one at a time, each walks a ladder of reductions from smallest to
    largest, and a reduction is kept only while it kept earning at least
    `min_gain` of the baseline error. The first step that stops paying ends that
    technology's ladder, so lifetimes are shortened as far as the fit rewards
    and no further.

    There is one roster per technology, not one per batch. Each pass ranks
    what is left against the CURRENT fit, takes the single worst offender, walks
    its ladder, and re-ranks. This matters because shortening a lifetime frees
    that technology's stock into the competition and whichever technologies
    absorb it can move a long way: a batch of candidates picked from one roster
    is scored against a fit that stopped being true after the first ladder in the
    batch. Re-ranking between ladders means every choice after the first is made
    on current information, and a technology that becomes an offender only
    because of an earlier reduction gets its turn.

    Technologies that have already walked a ladder are reported in later rosters
    but never re-opened: their ladder ended where it stopped paying, and
    re-opening it against a fit their own reduction produced is how a search like
    this talks itself into shortening everything at the node. The loop stops when
    a roster nominates nobody, when a ladder keeps nothing (the fit is unchanged,
    so the next roster would repeat the current one), or after `max_techs`
    ladders.

    Note that lifetime has two effects, both pushing share down. It sets the
    retirement rate — the intended lever — and, wherever `capital_recovery` is not
    defined at the technology, it is also the payback period in the capital
    recovery factor, so a shorter lifetime raises the annualised capital cost. The
    FIC fit absorbs part of that second effect, but a fitted lifetime is not a
    pure retirement statement.

    Parameters
    ----------
    lifetime_ladder : sequence of float
        Multipliers on the original lifetime, tried in order.
    lifetime_min : float
        Floor on the fitted lifetime, in years. A modelling judgement, not
        something the fit can tell you — set it to what the technology could
        plausibly last. Defaults to 3, low enough to stay out of the fit's way;
        raise it where a shorter life than that is not credible for the node.
    min_gain : float
        A reduction has to cut total L1 error by at least this fraction of the
        baseline to be kept. This is the "substantially" in the rule above.
    max_techs : int
        How many ladders to walk, i.e. how many roster/ladder passes to run at
        most. Each pass nominates one technology. Technologies that hold stock but
        are outside their availability window for the whole fitted range are
        nominated first, because retirement is the only lever that touches them
        at all — `_find_competing_weights` drops them from the competition, so no
        FIC has any effect. The rest follow, worst overshoot first. With
        `verbose`, the whole competition is printed at every pass, with the reason
        beside every technology that is not nominated.
    plot : bool
        Plot the node's market-share series at every step — the baselines, each
        rung of every ladder, and the final fit — via
        `plot_ms_for_node.plot_ms_line`. Each call renders its own figure, titled
        with the step it belongs to, so nothing is overwritten; the figures are
        also returned as `result['figures']`. Off by default: a fit at a
        thirteen-technology node walks a dozen rungs, and that is a dozen
        figures.

        Figure order is: the BEFORE baseline; a second BEFORE baseline at the
        final settings, only when `search_kwargs` is set to something different
        from `fit_kwargs`; then one figure per ladder rung, labelled with the
        pass, the technology, the lifetime being tried and the reductions kept so
        far; then the AFTER fit. The baselines come first deliberately — they are
        the untouched model, and a picture of it after every ladder figure reads
        as a step that undid the progress.
        Note also that the last rung of a ladder is normally the REJECTED one:
        the ladder stops at the first rung that does not pay, so the final fit
        matches the second-to-last rung, not the last.
    plot_kwargs : dict or None
        Passed through to `plot_ms_line` — `techFilters` in particular, which is
        worth setting when the node has enough technologies to make the legend
        unreadable.
    plot_aggregate : bool
        Run `aggregation_traversal` on the step's model before plotting. The
        market shares themselves are already on the graph; this fills in the
        aggregated quantities. Set False if it is not needed at your node, since
        it is not free.
    min_decline_run : int
        How many consecutive years the counterfactual has to go without rising
        before a technology is considered at all. This is the "is a shorter
        lifetime even the right lever" test: a counterfactual that keeps climbing
        is not asking for faster retirement, whatever the endpoints do. Flat
        counts as non-rising, so a technology the counterfactual holds at zero
        qualifies here. Set it to 0 to skip the test and judge on overshoot
        alone.
    min_overshoot : float
        Mean excess over the counterfactual, measured from the first year of that
        stretch to the end of the range, needed to qualify.
    peak_overshoot : float
        A single-year excess this large qualifies a technology on its own, even
        when the mean is below `min_overshoot` — a short, large miss is still a
        miss. Lower it to catch more, raise it to insist the overshoot be
        sustained.
    search_kwargs : dict or None
        Settings for the fits used to SEARCH — ranking candidates and scoring
        every rung of every ladder. Defaults to None, meaning **use
        `**fit_kwargs`**: the ladder is scored against the same objective the
        answer is fitted with, so a rung's gain means the same thing as the final
        number and there is only one baseline to compare against.

        Pass a dict to override. `search_kwargs={}` searches at the module's fast
        defaults, which is meaningfully quicker — `ridge` alone costs 1.5-2.5x
        the run time, and a ladder is many full fits — at the price of ranking
        lifetimes under one objective and reporting them under another. That is a
        reasonable trade on a large node when the ranking is all that is wanted,
        but it can select different technologies than the consistent search does.
    **fit_kwargs
        Settings for the FINAL fit, once the lifetimes are chosen — `ridge`,
        `smooth`, tolerances, and so on. These shape the FICs you keep.

    Returns
    -------
    dict with 'lifetimes' (tech -> fitted lifetime, original where unchanged),
    'baseline' (search-settings error before any lifetime change), 'final'
    (fit_kwargs error after), 'final_baseline' (fit_kwargs error before, so the
    improvement is like-for-like), 'fit' (the final per-year FIC result),
    'trials' (every reduction tried and what it scored, at the search settings, each
    tagged with the pass it came from), 'rosters' (one table per pass — every
    technology in the competition with its overshoot, whether it was nominated
    and why not, as `verbose` prints them), 'figures' (every plotly figure made
    when `plot` is on, in render order), 'roster' (the first of those tables, the
    ranking against the untouched model) and 'candidates' (its eligible subset).
    """
    import pickle as _pickle

    all_techs = node_info.list_techs(model.graph, nodeName)
    all_years = node_info.list_years(model.graph, nodeName)
    original = {t: model.get_param(PARAM.lifetime, nodeName, all_years[0], tech=t)
                for t in all_techs}

    snapshot = _pickle.dumps(model, -1)
    # Default: search at the SAME settings as the final fit. Passing a dict
    # overrides that — `search_kwargs={}` restores the module's fast defaults for
    # the ladder, which is quicker but scores the rungs against a different
    # objective from the one the answer is fitted with.
    search_kwargs = dict(fit_kwargs) if search_kwargs is None else dict(search_kwargs)
    same_settings = search_kwargs == fit_kwargs
    figures = []

    def fit_with(lifetimes, target_model=None, kwargs=None, label=None):
        m = target_model if target_model is not None else _pickle.loads(snapshot)
        for tech, value in lifetimes.items():
            if value is not None and value != original[tech]:
                _set_lifetime(m, nodeName, tech, value, all_years)
        result = optimize_ms_via_fics_v2(m, nodeName, verbose=False,
                                         **(search_kwargs if kwargs is None else kwargs))
        if plot and label:
            # `m` is the fitted model for this step, so the plot shows exactly
            # what the number beside it was computed from.
            _plot_step(m, nodeName, label, plot_kwargs, figures,
                       aggregate=plot_aggregate, verbose=verbose)
        return sum(r['end'] for r in result.values()), result

    if verbose:
        print(f"fitting the baseline (original lifetimes, "
              f"{fit_kwargs if same_settings else search_kwargs})...")
    baseline, baseline_fit = fit_with(
        {}, label='BEFORE: original lifetimes, '
                  + ('final settings' if same_settings else 'search settings'))
    if verbose:
        note = ('' if same_settings
                else '   (search settings; the final fit uses the arguments you passed)')
        print(f"baseline fit L1 = {baseline:.4f}{note}")

    # The search runs at `search_kwargs`; the answer is fitted at `fit_kwargs`, and
    # the "before" the improvement is measured against has to be the UNCHANGED
    # model at those same settings — otherwise the comparison mixes two different
    # objectives. It does not depend on the ladder, so it is computed here rather
    # than at the end: with `plot` on, that puts its figure beside the other
    # baseline instead of after every ladder figure, where a picture of the
    # untouched model reads as a step that undid all the progress.
    final_baseline = (baseline if same_settings else
                      fit_with({}, kwargs=fit_kwargs,
                               label='BEFORE: original lifetimes, final settings')[0])
    if verbose and not same_settings:
        print(f"baseline at the final settings {fit_kwargs} = {final_baseline:.4f}"
              f"   (the 'before' the improvement is measured against)")

    def roster_from(fit_result, exclude=()):
        # max_techs=1: each roster nominates only the worst remaining offender.
        return _lifetime_roster(fit_result, all_techs, model, nodeName,
                                lifetimes=original, max_techs=1,
                                min_decline_run=min_decline_run,
                                min_overshoot=min_overshoot,
                                peak_overshoot=peak_overshoot,
                                exclude=exclude)

    threshold = min_gain * baseline
    accepted = {}
    trials = []
    rosters = []
    walked = set()
    best, best_fit = baseline, baseline_fit

    def walk(entry, rnd):
        """Walk one technology's ladder; returns True if a reduction was kept."""
        nonlocal best, best_fit
        tech = entry['tech']
        walked.add(tech)
        if original[tech] is None:
            return False
        kept = None
        for mult in lifetime_ladder:
            value = max(original[tech] * mult, lifetime_min)
            if kept is not None and value >= kept:
                break                      # floor reached, no further reduction
            trial = dict(accepted)
            trial[tech] = value
            if verbose:
                print(f"   try {tech[:40]:<42} lifetime {original[tech]:g} -> {value:5.1f}"
                      f"   (kept so far: {_kept_note(accepted, original)})")
            score, result = fit_with(
                trial, label=f"pass {rnd} rung: {tech} lifetime "
                             f"{original[tech]:g} -> {value:g}"
                             f"  (kept so far: {_kept_note(accepted, original)})")
            gain = best - score
            trials.append({'tech': tech, 'lifetime': value, 'fit': score,
                           'gain': gain, 'stock_only': entry['stock_only'],
                           'round': rnd})
            if verbose:
                print(f"       -> fit {score:.4f}   gain {gain:+.4f}"
                      f"   {'keep' if gain >= threshold else 'stop'}")
            if gain < threshold:
                break                      # this step did not pay; stop shortening
            kept, best, best_fit = value, score, result
        if kept is not None:
            accepted[tech] = kept
            return True
        return False

    # One roster per technology, not one roster per batch. Each pass ranks what
    # is left against the CURRENT fit, takes the single worst offender, walks its
    # ladder, and re-ranks. Shortening a lifetime frees that technology's stock
    # into the competition, and whichever technologies absorb it can move a long
    # way — a batch of candidates chosen from one roster is scored against a fit
    # that stopped being true after the first ladder in the batch. Re-ranking
    # between ladders means every choice after the first is made on current
    # information, and a technology that only becomes an offender because of an
    # earlier reduction gets its turn.
    #
    # Technologies that have already walked a ladder are excluded from later
    # rosters: their ladder ended where it stopped paying, and re-opening it
    # against a fit their own reduction produced is how a search like this talks
    # itself into shortening everything at the node.
    for rnd in range(1, max_techs + 1):
        roster_n = roster_from(best_fit, exclude=walked)
        rosters.append(roster_n)
        candidate = next((r for r in roster_n if r['tested']), None)
        if verbose:
            if rnd == 1:
                _print_roster(roster_n, nodeName)
            else:
                print(f"\nroster check {rnd} — re-ranked against the current fit "
                      f"(technologies already walked are not re-opened):")
                _print_roster(roster_n, nodeName)
        if candidate is None:
            break
        if not walk(candidate, rnd) and verbose:
            print(f"   no reduction paid for {candidate['tech']}; "
                  f"the fit is unchanged, so re-ranking would repeat this roster")
            break
    else:
        if verbose:
            print(f"\nstopped after {max_techs} ladders (max_techs)")

    roster = rosters[0] if rosters else []
    lifetimes = {t: accepted.get(t, original[t]) for t in all_techs}

    if verbose:
        print(f"\nfinal fit with the arguments you passed: {fit_kwargs}"
              f"   ({len(accepted)} lifetime change(s))...")
    final, final_fit = fit_with(accepted, target_model=model, kwargs=fit_kwargs,
                                label='AFTER: fitted lifetimes, final settings')

    if verbose:
        print(f"final fit L1 = {final:.4f}  (baseline {final_baseline:.4f}, "
              f"{100 * (final_baseline - final) / final_baseline:.1f}% better)")
        for tech, value in accepted.items():
            print(f"   {tech:<46} lifetime {original[tech]:g} -> {value:g}")

    return {'lifetimes': lifetimes, 'changed': accepted, 'baseline': baseline,
            'final_baseline': final_baseline, 'final': final, 'fit': final_fit,
            'trials': trials, 'roster': roster, 'rosters': rosters,
            'figures': figures,
            'candidates': [r for r in roster if r['eligible']]}
