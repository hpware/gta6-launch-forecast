"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Each case gives hour 1's print and the hourly curves of past big game
launches around their own release day.  A launch day has a shape that
repeats across games - the same staggered midnight unlocks, the same
global evening peak - so the forecast scales that shape from the
comparables by this launch's first hour and keeps the spread the
comparables really show.

Quantiles are log-normal around the comparable mean: q = p1 * exp(mu + z*sd).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The levels the case asks for, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level above.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

HOURS = list(range(2, 25))
# A comparable needs a real launch hour to be a useful shape: smaller
# categories are sampled sparsely and their ratios are mostly noise.
MIN_FIRST_HOUR = 10_000
# Ratios beyond this are a launch that did not start in hour 1; clamping
# keeps a handful of them from dragging the whole ensemble.
RATIO_CAP = 20.0
# Widen the comparable spread a little: this launch is one draw, not the mean.
SPREAD = 1.05
# A floor under the log spread, from hour to hour.
FLOOR = 0.07


def parse_time(text):
    """An ISO-8601 UTC instant as a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hourly_curve(comparable):
    """The comparable's viewers for each hour 1..24 of its release day."""
    day = parse_time(comparable["day_start"])
    buckets = {}
    for point in comparable.get("hourly") or []:
        when = parse_time(point["from"])
        hour = int((when - day).total_seconds() // 3600) + 1
        viewers = point.get("viewers")
        if 1 <= hour <= 24 and viewers is not None:
            buckets.setdefault(hour, []).append(viewers)
    return {hour: sum(v) / len(v) for hour, v in buckets.items() if v}


def usable(comparable, name):
    """Whether a comparable can shape the forecast, and its hour-1 print."""
    if comparable.get("name") == name:
        return None
    # Early-access launches put part of the audience on another day.
    if "early-access" in (comparable.get("source") or "").lower():
        return None
    curve = hourly_curve(comparable)
    first = curve.get(1)
    if not first or first < MIN_FIRST_HOUR:
        return None
    if sum(1 for hour in HOURS if hour in curve) < 12:
        return None
    return curve


def launch_shape(case):
    """Per hour, the mean and spread of log(viewers/hour 1) over launches."""
    name = case["game"]["name"]
    ratios = {hour: [] for hour in HOURS}
    for comparable in case.get("comparables") or []:
        curve = usable(comparable, name)
        if curve is None:
            continue
        first = curve[1]
        for hour in HOURS:
            viewers = curve.get(hour)
            if viewers and viewers > 0:
                ratio = math.log(viewers / first)
                ratios[hour].append(max(-math.log(RATIO_CAP), min(math.log(RATIO_CAP), ratio)))
    shape = {}
    for hour in HOURS:
        values = ratios[hour]
        if len(values) < 3:
            shape[hour] = (0.0, 0.35)
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        shape[hour] = (mean, max(math.sqrt(variance), FLOOR) * SPREAD)
    return shape


def first_hour(case):
    """Hour 1's print, never below 1 viewer."""
    for point in case.get("known") or []:
        if point.get("hour") == 1:
            return max(1.0, float(point.get("viewers") or 0))
    raise ValueError("the case has no hour-1 print")


def forecast(case):
    """One row per forecast hour: the five quantiles of that hour's viewers."""
    first = first_hour(case)
    shape = launch_shape(case)
    rows = []
    for hour in HOURS:
        mean, spread = shape[hour]
        quantiles = []
        for weight in Z:
            value = first * math.exp(mean + weight * spread)
            value = min(1_000_000_000.0, max(1.0, value))
            quantiles.append(round(value, 2))
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
