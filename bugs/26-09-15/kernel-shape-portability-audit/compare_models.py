"""Compare complete saved logits and ordinary production samples."""
import argparse
import json
import statistics
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
records = [json.loads(path.read_text()) for path in (args.baseline, args.candidate)]
comparison = []
for stage in ('prefill', 'decode-1', 'decode-2'):
    samples = [[r['elapsed_ns'] for r in data if r['stage'] == stage and not r['warmup']]
               for data in records]
    medians = [statistics.median(values) for values in samples]
    deviations = [statistics.median(abs(x - median) for x in values) / median * 100
                  for values, median in zip(samples, medians, strict=True)]
    logits = [np.load(path.with_name(path.stem + '-' + stage + '.npy'))
              for path in (args.baseline, args.candidate)]
    comparison.append(dict(stage=stage, before_median_ms=medians[0] / 1e6,
                           after_median_ms=medians[1] / 1e6,
                           change_percent=(medians[1] / medians[0] - 1) * 100,
                           before_mad_percent=deviations[0], after_mad_percent=deviations[1],
                           identical_logits=bool(np.array_equal(*logits)),
                           max_abs_logits=float(np.abs(logits[0] - logits[1]).max())))
args.output.write_text(json.dumps(comparison, indent=2) + '\n')
print(args.output.read_text())
if any(not r['identical_logits'] or r['change_percent'] > 5 or
       max(r['before_mad_percent'], r['after_mad_percent']) > 5 for r in comparison):
    raise SystemExit('Comparison requires investigation under the declared protocol')
