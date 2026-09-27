#!/usr/bin/env python3
"""Write explicit renamed-component fixtures for `bench --input`, not search-study.

Every copy has the same four Boolean choices, two exactly-one constraints and
one complete cost table. The independent oracle enumerates its 16 assignments.
These isolate reusable structure; they are not hardware models or Qwen kernels.
"""
import argparse
import itertools
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
tuples = list(itertools.product(range(2), repeat=4))
entries = [[list(t), (sum(v << i for i, v in enumerate(t))*17) % 23 + 1] for t in tuples]
optimum = min(cost for t, cost in entries if sum(t[:2]) == 1 and sum(t[2:]) == 1)
for copies in (1, 8, 32, 128, 512):
    variables, factors = [], []
    for copy in range(copies):
        ids = list(range(4*copy, 4*copy+4))
        for i in ids:
            variables.append(dict(name=f'occurrence-{copy}-choice-{i}', domain=dict(runs=[dict(first=0,last=1,step=1)])))
        for pair in (ids[:2], ids[2:]):
            factors.append(dict(guards=[],kind={'Constraint':{'ExactlyOne':{'variables':pair}}}))
        factors.append(dict(guards=[],kind={'Cost':{'Table':dict(variables=ids,entries=entries)}}))
    record = dict(schema=1,name=f'renamed-components-{copies}',family='renamed-components',seed=0,
        parameters=dict(n=copies,d=2,width=4,depth=2,repeat=1,capacity=0,horizon=0,outputs=0,input_length=0,overhead=0,element_bytes=1),
        model=dict(variables=variables,factors=factors,units='synthetic cost'),
        reference={'Exact':copies*optimum},notes='Independent copies; exhaustive 16-assignment per-copy oracle. Distinct execution costs.',generation_ms=None)
    (a.out/f'renamed-components-{copies}.instance.json').write_text(json.dumps(record,indent=2)+'\n')
