# Fixed Metal interaction study

Exploratory research, not production characterization or a certified error model. Read the [protocol](../../../../specs/26-09-21/metal-interaction-study.md) and [findings](../../../../specs/26-09-21/metal-interaction-findings.md).

No production candidates are timed. Native FMA probes measure an execution mechanism; Seismic strict arithmetic is separately exercised using the actual repository helper source. Every generated manifest and raw observation belongs under ignored `results/`. Do not stage generated outputs.

## Reproduce

Local Python requires NumPy, SciPy and Matplotlib. The measuring Mac requires Swift Command Line Tools and Metal runtime support. The recorded study ran on an Apple M4 Pro in a fresh isolated directory (`mktemp -d`).

From this directory:

```sh
python3 generate.py
```

Copy `runner.swift`, `archive.swift` and `results/sources/*` into a new isolated directory on the chosen host. Within that remote directory:

```sh
xcrun swiftc -O runner.swift -o runner
./runner . cal warm-first --warm > cal-warm-first.jsonl 2> cal-warm-first.stderr
```

Copy the calibration bundle into local `results/`. Freeze predictions **before** collecting held-out measurements:

```sh
python3 analyze.py freeze
```

`freeze` refuses to overwrite `results/frozen-predictions.json`. Preserve the existing experiment in a separate results directory before starting a new study; do not overwrite the recorded freeze to fit new outcomes.

Then run on the remote host:

```sh
./runner . test warm-test --warm > test-warm.jsonl 2> test-warm.stderr
./runner . diagnostic warm-diagnostic --warm > diagnostic-warm.jsonl 2> diagnostic-warm.stderr
./runner . all warm-repeat --warm > all-warm-repeat.jsonl 2> all-warm-repeat.stderr
```

Collect GPU frequency telemetry separately with `sudo powermetrics --samplers gpu_power -n 90 -i 1000`, retaining stdout. Avoid other GPU workloads. Warm-up is a declared experimental condition, not an enforced fixed GPU frequency. The same input buffers are rewritten before each case; memory observations must not be interpreted as isolated named-cache measurements.

Copy bundles to local `results/`, then:

```sh
python3 analyze.py assess test-warm.jsonl
python3 analyze.py assess diagnostic-warm.jsonl
python3 analyze.py assess all-warm-repeat.jsonl
python3 followup.py
```

The follow-up is a **new diagnostic corpus**, motivated by the original held-out outcomes. Copy `results/followup/*` into a remote `followup/` directory and run:

```sh
./runner followup diagnostic interventions --warm > followup.jsonl 2> followup.stderr
xcrun swiftc -O archive.swift -o archive
./archive . > archives.log 2> archives.stderr
```

After copying `followup.jsonl` into local `results/`:

```sh
python3 summarize.py
```

## Files and limitations

- `generate.py`: fixed original source and manifest, exact repository helper extraction and identity.
- `runner.swift`: aggregate acquisition, warm-up, sampled CPU output checks, raw GPU/host timings, optional fixed-pipeline compiler limits. Host timing is submit-to-completion, excluding encoding, and may be amortized over repetitions.
- `analyze.py`: calibration-only freeze and unchanged held-out scoring. Diagnostic cases without a model are not counted as successful predictions.
- `followup.py`: exact input-multiset permutation, compiler-limit intervention, same-work/output lifetime intervention.
- `archive.swift`: observation-side fixed-kernel native archives. No native candidate facts enter predictions.
- `summarize.py`: reproducible descriptive statistics and static plot; no population or universal bound inferred from a sample maximum.

The current runner contains the follow-up pipeline support; captured original runner source is retained under results for exact historical provenance. One source compile failed initially because geometry attributes mixed scalar/vector types; the corrected source and failed compiler output are both retained. All 349 warmed original/intervention configurations completed, with sampled output checks passing. A G13-oriented disassembler could not decode the M4 archives; its output is not an instruction-count or register-allocation authority.

Native archives were extracted using `dougallj/applegpu` revision `4c5bae61086b8067231120c98b4756d7696d399c`, downloaded as source files over HTTPS into `results/tooling/`, not as a Git checkout. Its original license and failed disassembly are retained. No global tooling configuration was changed.
