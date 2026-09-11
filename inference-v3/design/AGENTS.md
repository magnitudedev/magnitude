This is NOT the top level "design" doc and does not participate in that system.

These docs are a HIGH LEVEL, HIGH SIGNAL source of truth for the v3 engine's
architecture: the boundary each component keeps, the guarantees it makes, why
they hold and what they cost. They constrain implementations; they do not
describe one.

Each document opens with one bold sentence stating what its component contains
and guarantees. Diagrams show flows and structure; tables state rules with their
reason or trade-off; a text block shows a worked example where a rule is easier
seen than said. Prose is for a boundary that needs explaining.

Do not describe the code. Do not enumerate modules, classes or functions, paste
signatures, or restate docstrings. Do not add benchmark numbers, hardware
inventories, progress reports, investigation notes or validation logs; those
belong in run records under `results/` and `runs/`.

Do not include "applies_to" metadata, and do NOT mindlessly update these docs.

Link only between these docs, never to the project root design docs.

If there is a factual correction to make, it may be made autonomously.
Any structural change, document addition, or change that does not turn something
objectively false into something objectively true needs explicit user approval.
