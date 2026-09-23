# Validation generators

Fixture JSON is generated locally under ignored `results/fixtures/`. Only the
generators belong in source control. A clean checkout needs generation before
compiling tests that include fixtures:

```sh
uv run inference-v4/validation/generate_fixtures.py
uv run inference-v4/validation/generate_fixtures.py --test -- -p seismic-engine --test sampling --test vision_math --test sequence_program
```

The driver pins NumPy, uses the local `inference-v3` numerical references, and
generates all eleven codec, decoder, attention, recurrence, rotary, routing,
sampling, vision, and erf/GELU fixtures. No model download or GPU is required.
Vision and erf/GELU references use independent Python equations. `--source`
selects another V3 source directory; `--output` selects another output directory
when running generation without `--test`.

The driver completes every generator before replacing existing fixtures. The
individual reference generators record their source and generator hashes;
the erf/GELU array is produced by `qwen_vision_merger_reference.py --erf`.

Measurement reports also belong under ignored `results/`;
`v3_qwen_forward_bench.py` generates the full-model V3 measurements.
