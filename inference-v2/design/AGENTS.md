This is NOT the top level "design" doc and does not participate in that system.

These docs are meant as a HIGH LEVEL, HIGH SIGNAL source of truth.

Assembly sections contain only the component tree and supported numerical annotations.
Never add benchmark setup, hardware/shape inventories, provenance paragraphs, progress
reports, investigation notes, or explanations of missing results around the tree.
Those belong in existing benchmark records or session evidence, not design documents.

A benchmark citation must support the displayed percentage: matching measurement,
implementation, dimension, and an evaluated theoretical bound. Timing alone is not
an efficiency assessment. Never attach citations to `unresolved`/`unmeasured` or use
them to imply that hierarchy coverage is complete. Omit unavailable annotations;
keep the component in the tree. Resolve missing derivation/binding prerequisites
before running a campaign whose purpose is to fill efficiency percentages.

Do not include "applies_to" metadata, and do NOT mindlessly update these docs.

Link only between these docs to each other, never to the project root design docs.

If there is a factual correction or change to make, it may be made autonomously.
Any significant structural changes, document additions, or any change that doesn't change something objectively false to something objectively true should not be conducted without explicit user approval.
