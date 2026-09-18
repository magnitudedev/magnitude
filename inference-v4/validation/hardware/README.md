# Independent Metal mapping evidence

These tools qualify hardware and native mappings outside compiler selection.
Their outputs are observations, not production hardware profiles. Write generated
JSON, captures and measurements under `../results/hardware/` (ignored). Commit
the tools and commands only; never add generated reports to Git.

`metal_services.swift` records timed arithmetic, copy, and matrix fixtures.
`qualify_services.py` computes slopes and retains their unqualified status:
loop control, native transformations, and compound conversions prevent treating
source operation counts as native instruction counts.
Passing `OUTPUT.json --archive-directory DIRECTORY` to the compiled probe also
captures each measured pipeline's source and native archive. Every observation
references its pipeline identity; archive/source hashes and public pipeline facts
are stored in the same report.

`metal_archive.swift` captures a runtime-compiled native archive even when the
command-line Metal compiler is unavailable. It uses the production runtime's
`fastMathEnabled = false` setting and records source/archive hashes, OS/device
identity, and public pipeline facts. It does not run or rank Seismic candidates.

```sh
swiftc -O metal_archive.swift -o metal-archive
./metal-archive arithmetic_mapping.metal max1 output/max1
./metal-archive arithmetic_mapping.metal max8 output/max8
```

The same fixture supplies `fma1` and `fma8`. Each capture retains its source,
native archive, and JSON report. Pipeline thread limits do not determine
resident-group capacity, register allocation, or spills.

## M4 Pro observation, 2026-09-18

`../results/hardware/m4-pro-native-mapping-2026-09-18.json` identifies all artifacts and the pinned
third-party extractor. The extracted `_agc.main` bytes for `max1` and `max8`
are identical (120 bytes each); the FMA variants differ (122 and 182 bytes).
A fixed positive cost per source max occurrence cannot describe both fixtures.
This is native byte-comparison evidence, not a timing or numerical qualification.

The pinned applegpu G13/M1 decoder cannot reliably decode these M4 binaries.
All four have failed decode markers, and `fma8` raises an assertion. Its
mnemonics and inferred resource counts must not be used. Native inspection
still needs a decoder appropriate to the target or another independently
validated interpretation. No complete qualified hardware objective exists yet.

## M3/M4 decoder and linked timing captures

The upstream [M4 issue](https://github.com/dougallj/applegpu/issues/62) points to
the [M3 branch](https://github.com/TellowKrinkle/applegpu/tree/b317f20bbf308959fa7d36246352070a8543f15b).
That pinned revision decodes the four arithmetic fixture bodies without reported
decode failures. It identifies one and eight FMA instructions in the FMA loop
bodies and one max instruction in both max loop bodies. This supersedes the
decoder-availability limitation above, not the failed G13 results or qualification
requirements.

`inspect_archives.py` checks the captured source/archive hashes before extraction
and joins native inspection with the actual timing observations. It accepts
explicit paths to an external extractor and decoder, records their identities,
and withholds mnemonic counts on decode errors. It distinguishes partially named
instructions and static main-shader counts from dynamic execution counts.

```sh
./metal-services measurements.json --archive-directory archives
python3 inspect_archives.py measurements.json archives /path/to/extractor \
  /path/to/M3/disassemble.py inspection.json
```

`../results/hardware/m4-pro-services-native-2026-09-18.json` retains 130 observations from 34 captured
pipelines. The pinned decoder handles 28 bodies without reported errors; all six
matrix bodies fail operand decoding (`Bad register size 3`). Scalar values were
checked for finiteness, while the uniform matrix fixture was checked against its
exact expected value. Scalar numerical qualification is still missing. Division
and transcendental bodies show compound instruction sequences, and some decoded
fields remain partially understood. None of these observations establishes a
complete primitive timing contract, resource occupancy, or held-out prediction.

## Scalar numerical observations

`metal_services.swift` now compares every scalar output against a stepwise host
Float reference, with explicit BF16/FP16 rounding and Darwin math functions.
It records absolute, relative and ULP error, nonfinite and unequal-bit counts,
and raw input/expected/actual bits for all 31 distinct inputs. These comparisons
do not choose a tolerance or qualify arbitrary inputs.

`../results/hardware/m4-pro-services-numerical-2026-09-18.json` records a fresh 34-pipeline,
130-observation run and the location/hashes of its complete report and archives.
All main shader bodies are byte-identical to the earlier mapping capture.
Most scalar chains match the host references bit-for-bit; logarithm and sine
chains reach maximum relative errors of approximately `8.37e-5` and `7.53e-5`,
respectively. Cosine differs by at most two ULPs. No scalar output is nonfinite.
The input set is positive and small, and the conversion chains quickly become
stationary. Matrix fixtures still use uniform exact data, and all six matrix
native decodes still fail. These results improve numerical evidence for the
observed fixtures, not full-model qualification or timing-model validity.

## Held-out chain prediction

`--held-out` measures body counts 2/4 instead of calibration counts 1/8, with
different loop lengths (scalar 512/2048, matrix 128/512). Numerical checks and
native archive capture remain enabled. `predict_chains.py CALIBRATION HELD_OUT
OUTPUT` fits only calibration data to `dispatch + iterations * (loop + count *
operation)`, then reports every held-out error without admitting any coefficient
as a hardware contract. It rejects overlapping calibration/test cases and
different device/OS identities. The fit is a deliberately limited hypothesis.

The M4 Pro result is `../results/hardware/m4-pro-heldout-chains-2026-09-18.json`: 120 predictions,
38.8% median relative error, 225.3% maximum, and only eight predictions inside
the observed three-sample ranges. Control kernels remain within roughly 1% of
their earlier medians, but copy kernels take 2.07–2.92 times as long. Device
operating-state variation is therefore a confounder; these errors cannot all be
assigned to native compiler transformations. Neither this additive hypothesis
nor the previous raw slopes justify a production timing contract.

Full held-out measurements and archives remain on `m4-pro-01` under
`~/seismic-v4-validation-01a0b3ad/metal-services-heldout.json` and
`heldout-service-archives/`; the driver is `metal_services_heldout.swift`.
Calibration uses the preceding `metal-services-numerical.json` capture.
