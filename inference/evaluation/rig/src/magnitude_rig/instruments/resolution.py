"""What a set of samples can resolve: the median and the half-width of its 95% interval.

The interval is distribution-free: the order statistics at ranks n/2 ± 1.96·√n/2, so it holds
for skewed timing samples. With fewer than six samples it is the full range.
"""

import math
import statistics


def median_interval(samples: list[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    n = len(ordered)
    middle = statistics.median(ordered)
    if n < 6:
        return middle, (ordered[-1] - ordered[0]) / 2
    reach = 1.96 * math.sqrt(n) / 2
    low = ordered[max(0, math.floor(n / 2 - reach))]
    high = ordered[min(n - 1, math.ceil(n / 2 + reach))]
    return middle, (high - low) / 2
