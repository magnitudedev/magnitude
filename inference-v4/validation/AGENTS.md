# Generated validation artifacts

Commit generator source and reproducible commands, never generated fixtures,
JSON reports, hardware measurements, or captured outputs. Do not force-add
ignored artifacts. Generated output belongs under `results/`, which is ignored.

`generate_fixtures.py` must cover every fixture consumed by engine tests. Run it
before compiling those tests, or use its `--test -- <cargo arguments>` wrapper.
Keep the numerical reference independent of the implementation being tested.
