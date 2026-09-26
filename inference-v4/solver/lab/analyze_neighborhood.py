#!/usr/bin/env python3
"""Paired diagnostics from saved study records; no additional optimizer runs."""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path


def target(record):
    ref = (record.get('reference') or {}).get('outcome')
    if not isinstance(ref, dict) or 'Optimal' not in ref:
        return None
    optimum = ref['Optimal']
    return next((p for p in record['points'] if p['elapsed_ms'] <= record['settings']['milliseconds']
                 and p['cost'] is not None and p['cost'] * 100 <= optimum * 105), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cohort', type=Path)
    args = parser.parse_args()
    groups = defaultdict(dict)
    for path in sorted((args.cohort / 'runs').glob('*.study.json')):
        record = json.loads(path.read_text())
        key = (record['family'], json.dumps(record['parameters'], sort_keys=True))
        groups[key][(record['instance'], record['seed'], record['method'])] = record
    results = []
    for (family, parameters), records in groups.items():
        ids = sorted({(i, s) for i, s, _ in records})
        for method in ['exact', 'random', 'anneal', 'joint', 'lns']:
            paired = [(records.get((i, s, 'greedy')), records.get((i, s, method))) for i, s in ids]
            paired = [(a, b) for a, b in paired if a is not None and b is not None]
            time_ratios, work_ratios = [], []
            baseline_reached = candidate_reached = 0
            for a, b in paired:
                at, bt = target(a), target(b)
                baseline_reached += at is not None
                candidate_reached += bt is not None
                if at is not None and bt is not None:
                    if bt['elapsed_ms'] > 0:
                        time_ratios.append(at['elapsed_ms'] / bt['elapsed_ms'])
                    if bt['work'] > 0 and method != 'random':
                        work_ratios.append(at['work'] / bt['work'])
            results.append({'family': family, 'parameters': json.loads(parameters), 'method': method,
                            'paired_runs': len(paired), 'greedy_reached': baseline_reached,
                            'method_reached': candidate_reached,
                            'both_reached': len(time_ratios),
                            'median_paired_time_speedup_over_greedy': statistics.median(time_ratios) if time_ratios else None,
                            'median_paired_work_speedup_over_greedy': statistics.median(work_ratios) if work_ratios else None,
                            'interpretation': 'Ratios condition on both reaching target; report both success counts. Shared-host wall time is observational. Random work units differ.'})
    (args.cohort / 'paired.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'wrote {len(results)} paired regime/method comparisons')


if __name__ == '__main__':
    main()
