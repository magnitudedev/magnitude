#!/usr/bin/env python3
"""Summarize the matched combined-pass datasets; raw records remain ignored."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
def read(path):
    return [json.loads(p.read_text()) for p in sorted(path.glob('*.json'))]
def median(values):
    return statistics.median(values)
def target(record, tolerance=1.05):
    optimum = record['reference']['outcome']['Optimal']
    return next((p['elapsed_ms'] for p in record['points']
                 if p['cost'] is not None and p['cost'] <= tolerance*optimum
                 and p['elapsed_ms'] <= record['settings']['milliseconds']), None)
def pairs(before, after):
    for p in sorted(after.glob('*.json')):
        a = json.loads(p.read_text())
        b = json.loads((before/p.name).read_text())
        assert a['fingerprint'] == b['fingerprint']
        assert a['options'] == b['options'] and a['settings'] == b['settings']
        assert a['status'] != 'error' and b['status'] != 'error'
        assert a['reference']['outcome'] == b['reference']['outcome']
        yield b, a

report = '''# Combined solver optimization measurements — 2026-09-18

All changes are evaluated together: structural completed-proof reuse,
shared repair knowledge/immutable factor definitions, reusable conflict proofs,
resource/critical-path move priorities and numeric thresholds, and stronger
resource propagation. No Seismic language, mathematical-model or public search
entry-point change. New counters expose proof stores, structural/conflict hits
and learned conflicts.

**Result: substantial gains on repeated structure and small schedules, not a
uniform speedup across all problems.** No whole-Qwen completed tuning estimate
is established; the compiler-export blockers reported in the Qwen qualification
remain separate.

## Repeated structure

Explicit copies of one four-variable problem, each with two exactly-one
constraints and a complete 16-row cost table. The independent per-copy oracle
enumerates all assignments, and total cost must be charged once per occurrence.
Three fresh-process runs per size/build, exact search, identical saved input
fingerprints. Median milliseconds. “Validation + solve” includes model cloning,
validation, search initialization, search and final witness checking; it excludes
JSON loading, source construction, process startup and lab metadata collection.

| Copies | Search before → after | Validation + solve before → after | Total speedup | Structural hits |
|---:|---:|---:|---:|---:|
'''
for n in (1, 8, 32, 128, 512):
    b = [r for r in read(root/'reuse-before/runs') if r['instance'] == f'renamed-components-{n}']
    a = [r for r in read(root/'reuse-after/runs') if r['instance'] == f'renamed-components-{n}']
    assert len(a) == len(b) == 3
    assert all(r['status'] == 'optimal' and r['cost'] == n*10 for r in a+b)
    assert {r['model_fingerprint'] for r in a} == {r['model_fingerprint'] for r in b}
    bs, ass = median(r['solve_ms'] for r in b), median(r['solve_ms'] for r in a)
    bt, at = median(r['solve_ms']+r['validation_ms'] for r in b), median(r['solve_ms']+r['validation_ms'] for r in a)
    report += f"| {n} | {bs:.3f} → {ass:.3f} | {bt:.3f} → {at:.3f} | {bt/at:.2f}× | {a[0]['stats']['structural_hits']} |\n"
report += '''
Search-only gains at 32 and 512 copies are approximately 7.5× and 26×.
Initialization/validation limits the overall gain. This isolates structural reuse;
these are not Qwen kernels, and matching real kernel signatures does not prove
matching complete boundary conditions. Recognition currently handles consistent
order-preserving variable renaming, not arbitrary graph isomorphism. The shared
store is scoped to one immutable model and its internal repairs; independent
public Search instances do not share a process-global cache.

## Small explicit schedules

Two four-activity variable-mode schedules, capacity two, horizon sixteen;
independent implementation/start enumeration gives optima four and eight.
Exact/joint, three optimizer seeds per input, one-second allowance. All 24
before/final runs attain the independent optimum. Joint correctly remains
incomplete when it lacks a global proof. Time to the first sampled optimum
includes instance construction, initialization and checked search witnesses.

| Method | Before median ms | Final median ms | Ratio |
|---|---:|---:|---:|
'''
sp = list(pairs(root/'scheduling/before/runs',root/'scheduling/final/runs'))
for method in ('exact', 'joint'):
    p = [(target(b,1),target(a,1)) for b,a in sp if a['method']==method]
    assert len(p)==6 and all(x is not None and y is not None for x,y in p)
    b,a=median(x for x,y in p),median(y for x,y in p)
    report += f'| {method} | {b:.3f} | {a:.3f} | {b/a:.2f}× |\n'
report += '''
The resource clique-bound algorithm now uses the threshold-demand graph's
structure in quadratic rather than cubic time. This is an algorithmic improvement
in that propagation step, not a quadratic bound on total search. Mandatory
positive-duration ordering and release-suffix energy/clique bounds also prune
more schedules. Independent small-schedule tests cover optional/zero-duration
activities and nonconvex domains.

## Broader matched sample

18 unchanged chain, contraction and shared-producer instances: sizes 8/32/64,
generator seeds 100–101, optimizer seeds 0–1, exact/joint. 72 runs per build,
1,000 ms including construction/initialization, same width/restart/options and
independent exact references. Columns show successful 5%-quality runs out of
four and median paired before/after time-to-target ratios; above one is faster.
Failed target runs are excluded from ratios but explicitly counted. Easy
contraction fixtures often meet the target at the initial sample, so small
ratios there primarily reflect overhead.

| Family | Size | Method | Before successes | After successes | Paired speedup |
|---|---:|---|---:|---:|---:|
'''
groups=collections.defaultdict(list)
pop=list(pairs(root/'before/runs',root/'final/runs'))
assert len(pop)==72
for b,a in pop: groups[(a['family'],a['parameters']['n'],a['method'])].append((target(b),target(a)))
for (family,n,method),p in sorted(groups.items()):
    ratios=[b/a for b,a in p if b is not None and a is not None]
    ratio=f'{median(ratios):.2f}×' if ratios else '—'
    report+=f"| {family} | {n} | {method} | {sum(b is not None for b,a in p)}/4 | {sum(a is not None for b,a in p)}/4 | {ratio} |\n"
report+='''
All joint runs retain target success. Exact still fails to find a witness on
32/64-consumer shared-producer instances within one second. Several cases are
slower: cache bookkeeping and heuristic changes are not free. This pass does
not establish a universal 2× improvement. No inference-speed improvement is
claimed; these changes affect optimization work.

## Interpretation for overall compilation

Reusable equivalent components are the strong result: 5.3× including solver
initialization at 32 copies is a useful measured mechanism. Its contribution to
whole compilation depends on how much currently uncached work has that shape.
If half of total compilation were accelerated 5.3×, overall speedup would be
about 1.68×; if 80% were accelerated, about 2.85×. These are conditional Amdahl
calculations, not Qwen forecasts. Existing compiler entry/workload caching may
already remove repeated requests. Source export, target model construction,
native compilation, and the first intrinsically difficult coupled solve remain.

## Reproduction, provenance and limitations

Recreate repetition inputs with `scripts/reuse_inputs.py --out DIR`, then run
both frozen binaries with `bench --input DIR --warmups 0 --repeats 3
--seconds-per-case 10 --out OUTPUT`. The broader study uses the saved `instances/`
and `search-study --methods exact,joint --search-seeds 0,1 --milliseconds 1000
--sample-work 100 --restart-after 18446744073709551615`. Scheduling uses
`scheduling/instances/`, seeds 0,1,2 and the existing default restart setting.
Every before/final study pair has identical input fingerprints and options.
The repeated-structure fixtures use `bench`, which reads their explicit models;
they must not be passed to `search-study`, which regenerates named families.

198 runs contribute to the final comparisons: 144 broad, 24 schedule, 30 reuse.
All available witnesses are checked by the lab; small cases have independent
oracles. No errors or reference contradictions appear in the final datasets.
Timeouts with no witness are retained as failures. The reference costs are
synthetic integer objectives, not accelerator timing.

The baseline is the preexisting frozen lab release executable, also used by the
previous neighborhood experiment; its source provenance/limitations are in that
report. This compares against that measured baseline, not an invented identical
source revision reconstructed after edits. Final measurements use one frozen
combined executable. Intermediate `after/` and `scheduling/after/` runs are
exploratory and superseded by `final/` and `scheduling/final/`.

Host: Apple M4 Max, 64 GB, release builds. Processes run sequentially within each
cohort; other development/tests were active, so small timing differences are
noisy. No statistical confidence interval or population-level guarantee is
claimed. Work units also changed cost and are not a wall-time substitute.
Raw JSON, binaries and logs remain ignored. Markdown reports are versionable.

Verification: 135 distinct core tests and 18 lab tests passed. The full core
run covered 132 tests; three additional targeted regressions cover narrower vs
wider conflict reuse, 250 shared-repair oracle runs, and outer cache-pressure
reclamation. Existing memory-resumption tests were rerun after that last fix.
The final review's cache-pressure reclamation fix postdates the frozen timed
binary; it affects only memory-pressure paths. Measured broad/scheduling runs
peaked below 19 MB against 512 MiB limits and had no memory stops; reuse runs
also stayed below their limits. Timing claims describe the frozen binary,
not an unmeasured rebuild. Public search/model semantics are unchanged.

'''
for name in ('baseline','combined-final'):
    report+=f'{name} SHA-256: `{hashlib.sha256((root/name).read_bytes()).hexdigest()}`.\n\n'
(root/'README.md').write_text(report)
print('Verified and summarized 198 matched before/final runs.')
