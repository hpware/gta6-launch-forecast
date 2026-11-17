"""Forecast each hour of a game launch day's Twitch viewers.

The case gives one launch's hour-1 print and the launch-day curves of past big
launches (comparables).  A launch day's shape is not the same at every scale:
a launch whose first hour is a huge spike keeps a smaller share of it later
than a smaller launch whose audience builds through the day.  So each hour's
ratio to hour 1 is read as a line through the comparables' own first-hour size,

    log(viewers(h) / viewers(1)) ~ a(h) + b(h) * log(viewers(1))

fitted locally around this launch's size, and blended with the comparables'
plain mean for a little robustness.  The residual spread of that fit, floored,
sets the five quantiles: q_tau(h) = p1 * exp(mu(h) + z_tau * sd(h)).

A launch bigger than every comparable is read at the largest one's size, never
extrapolated past it, and its spread is widened a little for that step outside
the data.  Deterministic: no clock, no randomness.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The scorable levels, in the order each row lists them, and the normal's
# quantile at each: a pinball loss wants the true quantiles.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

HOURS = list(range(2, 25))
# A comparable's first hour must be a real launch hour: small categories are
# sampled sparsely and their ratios are noise.
MIN_FIRST_HOUR = 10_000
# Ratios past this are a launch that did not really start in hour 1.
RATIO_CAP = 20.0
# How far, in log viewers, a comparable's weight reaches when the line is fitted.
BANDWIDTH = 0.5
# A floor weight keeps a launch far from this one's size from vanishing.
MIN_WEIGHT = 0.05
# How much of the plain comparable mean is mixed back into the fitted line.
MEAN_WEIGHT = 0.15
# A floor under the log spread, and a small widening for one launch's own draw.
FLOOR = 0.08
SPREAD = 0.9
# Extra log spread for every log viewer this launch is past the largest comparable.
OUTSIDE = 0.05


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hourly_curve(comparable):
    """The comparable's mean viewers for each hour 1..24 of its release day."""
    day = parse_time(comparable["day_start"])
    buckets = {}
    for point in comparable.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        when = parse_time(point["from"])
        hour = int((when - day).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            buckets.setdefault(hour, []).append(viewers)
    return {hour: sum(v) / len(v) for hour, v in buckets.items() if v}


def usable(comparable, name):
    """The comparable's launch-day curve, or None when it cannot shape this one."""
    if comparable.get("name") == name:
        return None
    if "early-access" in (comparable.get("source") or "").lower():
        return None
    curve = hourly_curve(comparable)
    first = curve.get(1)
    if not first or first < MIN_FIRST_HOUR:
        return None
    if sum(1 for hour in HOURS if hour in curve) < 20:
        return None
    return curve


def collect(case):
    """For each hour, the comparables' (log first hour, log ratio to it) points."""
    name = case["game"]["name"]
    points = {hour: [] for hour in HOURS}
    for comparable in case.get("comparables") or []:
        curve = usable(comparable, name)
        if curve is None:
            continue
        first = curve[1]
        x = math.log(first)
        for hour in HOURS:
            viewers = curve.get(hour)
            if viewers and viewers > 0:
                ratio = math.log(viewers / first)
                points[hour].append((x, max(-math.log(RATIO_CAP), min(math.log(RATIO_CAP), ratio))))
    return points


def weighted_mean(values, weights):
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def fit_hour(points, launch_size):
    """(centre, spread) of this hour's log ratio, for a launch of this size."""
    if not points:
        return 0.0, 0.5
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    if len(points) < 5:
        centre = sum(ys) / len(ys)
        spread = math.sqrt(sum((y - centre) ** 2 for y in ys) / len(ys)) if len(ys) > 1 else 0.3
        return centre, max(spread, FLOOR) * SPREAD
    low, high = min(xs), max(xs)
    target = max(low, min(high, launch_size))
    weights = [max(MIN_WEIGHT, math.exp(-abs(x - target) / BANDWIDTH)) for x in xs]
    mean_x = weighted_mean(xs, weights)
    mean_y = weighted_mean(ys, weights)
    spread_x = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, weights))
    slope = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in zip(xs, ys, weights)) / spread_x if spread_x > 0 else 0.0
    intercept = mean_y - slope * mean_x
    centre = intercept + slope * target
    residual = math.sqrt(sum(w * (y - intercept - slope * x) ** 2 for x, y, w in zip(xs, ys, weights)) / sum(weights))
    plain = sum(ys) / len(ys)
    mixed = (1.0 - MEAN_WEIGHT) * centre + MEAN_WEIGHT * plain
    outside = max(0.0, launch_size - high)
    return mixed, max(residual, FLOOR) * SPREAD + OUTSIDE * outside


def first_hour(case):
    for point in case.get("known") or []:
        if point.get("hour") == 1:
            return max(1.0, float(point.get("viewers") or 0))
    raise ValueError("the case has no hour-1 print")


def forecast(case):
    first = first_hour(case)
    points = collect(case)
    size = math.log(first)
    rows = []
    for hour in HOURS:
        centre, spread = fit_hour(points[hour], size)
        quantiles = []
        for z in Z:
            value = first * math.exp(centre + z * spread)
            quantiles.append(round(min(1_000_000_000.0, max(1.0, value)), 2))
        rows.append({"hour": hour, "q": sorted(quantiles)})
    return {"hours": rows}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(input_dir.glob("*.in"))
    if not cases:
        sys.exit(f"no case files in {input_dir}")
    for path in cases:
        case = json.loads(path.read_text(encoding="utf-8"))
        answer = forecast(case)
        (output_dir / f"{path.stem}.out").write_text(
            json.dumps(answer, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
