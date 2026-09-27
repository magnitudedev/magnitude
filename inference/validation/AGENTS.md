# Generated validation artifacts

Commit generator source and reproducible commands, never generated fixtures,
JSON reports, hardware measurements, or captured outputs. Do not force-add
ignored artifacts. Generated output belongs under `results/`, which is ignored.

If engine tests begin consuming generated fixtures, update `generate_fixtures.py`
to cover every consumed file and run it before those tests. The current engine
tests do not consume generated fixtures. Keep the numerical reference
independent of the implementation being tested.
