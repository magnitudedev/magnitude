# Component identification and assembly

**Component types own contracts and composable performance models. Implementations
realize those contracts. Assemblies select implementations; evidence establishes
what a particular revision achieves.** The [catalog](performance/catalog.md) links
each current type and dimension to its authoritative owner.

## Identifiers

```text
FAMILY:COMPONENT                         component type / contract
FAMILY:COMPONENT:SOURCE:VARIANT          implementation
FAMILY:COMPONENT/DIMENSION               performance dimension
```

Names are uppercase. Colons separate fields; dots express hierarchy within a
component address; underscores separate words within a name. Source and dimension
codes are concise, explicitly defined abbreviations.

| Field | Meaning | Examples |
|---|---|---|
| Family | Subsystem containing the component | `MODEL`, `SCHEDULING`, `KV` |
| Component | Responsibility within that family | `QWEN35.ATTENTION`, `ADMISSION`, `APPEND` |
| Source | Supplier of the implementation or owner of its composition | `MLX`, `LM`, `VLM`, `MAG` |
| Variant | Distinguishing implementation method | `PAGED`, `STANDARD`, `FIFO` |
| Dimension | Performance outcome defined by the type | `EXEC`, `MEM`, `RESTORE` |

```text
MODEL:QWEN35.ATTENTION
MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS
MODEL:QWEN35.ATTENTION/EXEC
MODEL:ATTENTION:MLX:DENSE
MODEL:FORWARD:VLM:STANDARD
STATE:QWEN35/MEM
STATE:QWEN35/RESTORE
```

Within `MODEL`, an architecture qualifier such as `QWEN35` or `GEMMA4` identifies
architecture-specific computation. An unqualified address such as `ATTENTION`
identifies shared model computation subject to its capabilities. Other families
use their own concepts; there is no universal generic/specific scope field.

## Vocabulary and uniqueness

| Family | Responsibility |
|---|---|
| `ENGINE` | Complete engine composition |
| `SCHEDULING` | Admission, service selection and allowances |
| `BATCHING` | Assembly of compatible ready work |
| `GENERATION` | Token advancement, drafting, verification and acceptance |
| `EXECUTION` | Device submission, completion and resource lifetime |
| `MEMORY` | Capacity accounting and reservations |
| `CACHE` | Reusable prefix indexing and retention |
| `STATE` | Logical model state and checkpoint composition |
| `KV` | Physical attention-history storage and operations |
| `MODEL` | Neural architectures and computational blocks |

Each type has one authoritative record in its owner document. Each implementation
ID names one selectable realization of that type. Definitions introduce new names;
other documents link to those definitions instead of introducing synonyms. Variants
describe methods, not rankings or benchmark results. `STANDARD` means conventional,
not automatically correct or preferred. Avoid opaque numbers and labels such as
`FAST`, `BEST` or `V2`.

## Identity and provenance

Source codes are `MLX` for MLX, `LM` for MLX-LM, `VLM` for MLX-VLM and `MAG` for
Magnitude. Source attribution follows composition ownership recursively:

- A Magnitude composition of upstream/owned pieces is `MAG`.
- An MLX-VLM composition of MLX operations is `VLM`.
- Unchanged pass-through preserves upstream identity; a substantive wrapper has
  its own separately identified composition.

Children retain their own sources. A parent does not concatenate codes or become
`MIXED`; neither does one owned child automatically relabel an upstream parent.
Technology is separate: an owned Metal kernel has source `MAG`, technology `MTL`
and may run through MLX. `MTL` is not a source code.

IDs name implementations, not loaded instances. Layers may share an ID while using
different weights. Artifact identity, configuration, dependency versions, hardware
and evidence attach to it. Ordinary improvements and file moves preserve the ID;
separately selectable methods get distinct variants. An incompatible contract needs
a distinct type rather than silently changing an old meaning.

Implementation fingerprints identify exact measured content and child selections.
Any implementation change resets current assessments through affected parent
compositions under the [evidence rules](performance.md#stable-compositions-and-evidence).
It does not erase history or rename the component.

## Blueprints and execution components

`@component("MODEL:ATTENTION:MAG:PAGED")` declares the canonical ID once on the
execution class or operation. The ID is validated and stored as a `ComponentId`.
Schemas, benchmarks and constructors reference that class or its instances; they
obtain the ID through `component_id(...)`. There is no separate ID catalog, source
argument, variant argument or reporting schema in the declaration.

`@blueprint` marks immutable construction instructions. Its `implementation()` already
references the constructor. A direct constructor exposes its component ID there; a
loader's produced execution objects expose their concrete identities after loading.
A blueprint does not copy IDs or construct a parallel hierarchy.

Capture follows actual children and shared dependencies, attaching the ID from each
object's declaration. A distinct execution child, such as compiled Qwen decode, carries
its own declaration. Deliberate implementations sharing an existing identity reference
its declaring class with `@component(ExistingClass)`; repeating an ID declaration fails.

Serialized graphs retain IDs and explicit edges across processes and machines. Code
moves preserve the declared ID; executable fingerprints track revisions separately.

Model roots also reference their production `ModelDefinition`, which owns both stable model
identity and default construction. Performance history uses that identity while recording
implementation changes as snapshots. [Performance bindings](performance.md#binding-the-executed-composition)
define how typed schemas read execution structure without adding reporting machinery to
numerical code.

## Definitions and assemblies

The [performance system](performance.md#ownership-and-component-records) defines
the consistent component record: contract, parameters, composition, dimensions,
implementations and controls. Contract boundaries include numerical behavior,
state/aliasing, supported inputs and resource lifetime. A reference identifies the
boundary and conditions it validates; naming one is not proof of equivalence.

Assemblies connect selected implementations and shared dependencies. Ownership and
sharing form a graph even when displayed as a tree. Semantic assemblies are documented for the
are the [engine](engine/components.md#assembly),
[Qwen](models/architectures/qwen35.md#assembly),
[Gemma](models/architectures/gemma4.md#assembly) and
[generic upstream executor](models/architectures/generic-mlx-vlm.md#assembly).
[Model composability](models/composability.md) defines substitution within those trees.

Architecture trees summarize the implementation's semantic structure with canonical IDs.
Show a repeated layer pattern once, distinguish alternatives from sequential work, and
identify optional branches and shared dependencies. Grouping labels are explanatory;
they do not invent component identities. Exact instance counts, paths and selected
bindings come from runtime captures in the performance tooling, which also owns current
assessments. A semantic group does not imply one percentage for all its instances.
Generated performance views use the [tree convention](performance.md#tree-annotations-and-benchmark-references).
Implementation identity, performance dimension and benchmark identity remain distinct;
the benchmark reference resolves to applicable recorded results without adding
provenance nodes to the assembly.

Matching types makes implementations comparison/substitution candidates; capabilities,
layouts and numerical contracts still have to match or be explicitly adapted.
Reference relationships are not production dependencies. Every identified boundary
must be independently exercisable, including outputs/state and performance properties.

A component boundary is not a mandatory object, registry lookup, Python call, GPU
launch or synchronization. Blocks may fuse/compile across boundaries. Test access
must not impose production intermediates; a fused region can be checked against
its composed contract. Performance propagation follows the demand/execution rules,
not a sum of isolated timings or percentages.
