---
applies_to:
  - inference/service/models/**
  - inference/service/server/src/assessment/**
  - inference/service/contracts/src/models.rs
  - inference/engine/src/assessment.rs
  - inference/engine/fixtures/assessment/**
  - inference/catalog/**
  - packages/icn/src/hardware/**
  - packages/icn/src/models/**
  - packages/acn/src/local-model*
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/local-models/**
---

# Model assessment and ranking

The inference service owns assessment-material resolution, assessment demand, filtering,
scheduling, cache reuse, concurrency, deadlines, and assessment publication. The inference engine
computes template-derived capabilities, compatibility, memory fit and performance. ACN owns catalog
ranking and projects assessment state. Clients only render the result.

Assessment Material is the immutable GGUF header evidence required to assess a model without
installed tensor payloads: each component's exact header, through the aligned tensor-data offset,
with its component roles and content identities. The header carries the tokenizer, chat templates,
metadata and tensor directory the engine reads; a companion projector header lets the engine
establish vision without an authored capability flag. Catalog desired, catalog effective, custom
model, and candidate targets resolve this same input shape. A catalog serving profile's context is the
target's supported maximum context, the context the engine serves and assesses; memory fit uses
`min(context, 100_000)`.

Temporary assessment files preserve the artifact's logical size without allocating its absent
tensor payload. Windows explicitly marks these files sparse before extending them; unsupported
filesystem operations fail that assessment rather than consuming model-sized disk space.

The automatic assessment pool assesses catalog desired material when not installed, effective material
when installed, and only `Ready` custom models with their attached drafter. It publishes a read-only
revisioned snapshot with independent catalog and custom source slices. Exact work identity guards publication, so removed
or superseded models cannot retain stale results. Packages, bundles, and serving configurations do
not cross the boundary. A failed source read keeps that slice as it is and is retried in the
background; it is the inventory's failure, which the catalog reports. Each exact target is attempted once; any target failure settles as `Dropped`, is never
retried, and is omitted by ACN. Catalog drops emit an OpenTelemetry error; custom-model drops are
silent. Candidate assessments are requested on demand for a candidate's target, projector, and
drafter, from headers fetched by range request, and are cached like every assessment.

One assessment reads the target's Assessment Material headers once, in the service process, and
is arithmetic over them and the assessment environment: its structure, established at start (the
selected device and the host), and what the measurement job has stored of this machine (the
selected device's costs, derived from its measured cost points). Until those exist a fitting model's
speed is pending; when they arrive every assessed model is finished again and published in one
revision. The engine's own tokenizer,
template and reasoning inspection supplies capabilities and the template fingerprint; its method
resolution decides speculative execution.
One deadline covers the target, and one flat assessed result
publishes capabilities, template fingerprint, and profile evidence together. There is no planning
worker, template worker, inventory capability state, or post-download assessment gate. Equal
immutable tokenizer and template content may share derived preparation within the process without
becoming a separate capability authority.

`Fits`, `DoesNotFit`, and `Unsupported` (custom models and candidates only) are genuine terminal
evidence.
Transport or operation failure drops the target rather than fabricating assessment evidence.
Hardware observations that were not performed are represented as `NotObserved`; zero-valued
headroom is never fabricated.

Ranking exists only for reviewed catalog models with `Fits` evidence. Intelligence and fidelity
come from authored catalog evidence; speed comes from engine assessment, at the bounded comparison
depth, once this machine is measured. A speed that is pending or unavailable yields an absent speed
score, never a zero: such a model is ranked by intelligence and fidelity alone, after every model
with a speed score, and never by name. Custom models are never ranked and never inherit an
intelligence or fidelity score from the catalog model they are a variant of.

Provider selection requires `Fits`, current selectability, profile, and capabilities from the same
assessed state. Package validation establishes only structural artifact validity and presence;
assessment is the sole published capability authority. Loading resolves the same engine
configuration from the installed package; the service verifies the loaded instance's template
fingerprint and modalities against the host's resolution. Admission plans memory against current
resources. Cached assessment alone never authorizes admission.

## Engine boundary

`ModelAssessment` is the one contract between the engine (families, weight types, kernels) and the
product (service, ACN, clients). The engine decides which outcome a file gets; the product asks and
presents. Adding a weight type or relaxing a family check changes verdicts, never the contract.

| Outcome | Carries |
|---|---|
| `Fits` | Profile, memory, performance, and the projector and drafter verdicts |
| `DoesNotFit` | Profile, memory, the limiting resource and deficit, and the projector and drafter verdicts |
| `Unsupported` | Profile when one exists, the complete, never empty list of `UnsupportedReason`s, and developer detail; an unreadable file has no profile |

The subject of an assessment is a catalog model (with its selection) or a custom model. A
candidate (a Hugging Face repository or a local file not yet added) is not a subject: a separate
candidate inspection call assesses its versions from range-fetched headers and returns the same
verdicts.

`UnsupportedReason` is a closed, structured set: architecture, weight types (each type with its
tensor count), rope scaling, unrecognized tensor, unrecognized metadata, missing metadata,
tokenizer, chat template, kernel domain, backend, unreadable, target mismatch, tied output head,
and drafter architecture. Target mismatch is a drafter (or projector) that does not fit the target it is bound to,
reported on that component's verdict: a projector the target's family can't bind (a text-only
family, another output width) is `ProjectorVerdict::Rejected` and the target is assessed and served
without it, taking text input only, so a projector never makes a target unsupported or unreadable;
drafter architecture is a separate drafter whose `general.architecture` no drafter family binds,
which names what the engine lacks rather than the target's architecture: a Nemotron-H MTP file
(`nemotron_h_moe`, an architecture the engine runs as a target) is refused this way, because the
engine drafts with no Nemotron-H prediction head, separate or built in;
missing metadata is a well-formed file that omits a key its family requires and never infers,
naming that key (a block drafter's attention causality, which some published drafters predate, or
a target's normalization epsilon); tied output
head is a target whose output projection is its embedding where the family needs a separate one. Every refusal names
at least one reason: a header missing a weight its family needs, or contradicting itself, is
unreadable, and a planner refusal of a validated definition names the backend. The assessment
collects every reason, not the first: a representation failure does not stop the weight-type
check, and a file with two unsupported types reports both. Clients own user-facing wording per
case; `detail` stays developer text for Details and `--json`. Clients render an unknown case
generically with its detail, so the engine can add cases without a client release. The engine
(ICN) updates independently of ACN, so the wire form is open: a reason ICN's contract doesn't name
decodes as an unrecognized reason keeping its tag and detail, and ACN projects it as `Unknown`. The
service never emits one itself.

Every readable candidate version gets the engine's verdict, an architecture the engine doesn't
recognize included; the product only labels, groups and orders by the engine's recognized
architecture sets and verdicts, and never judges a file by its name, its metadata key names or an
architecture list of its own. Built-in prediction heads are attached as chosen and bind or not by
the engine's drafter verdict.

`DrafterVerdict` is `None`, `Bound` (method and resident memory), or `Rejected` (reasons and
detail). A drafter that fails to bind is reported, never silently dropped. The engine reports only
whether a drafter binds; lineage stays a product judgment
(see [drafter selection](./drafter-selection.md)).

An assessed entry also carries the target's `ModelFingerprint`: architecture, decoder block count (without
MTP prediction layers, so a conversion that dropped the heads keeps it), embedding
length, head and KV-head counts, key length, vocabulary size, expert count, and a digest of the
tokenizer's token table. The engine also publishes the architecture strings it recognizes, which
Hugging Face search uses as its prefilter, and judges a model file's header alone
(`header_refusal`): the header interpretation admission runs (family, tokenizer, chat templates)
without weight types, a drafter or a device, or, from leading metadata only (a prefix), what those
entries settle by themselves (the architecture and the tokenizer scheme and flags they declare).
Search shows a result whose model file header is refused this way disabled with those reasons;
weight types never disable a search result, since another version may have supported ones. The product side never parses headers, reads file names
for quantization, or keeps its own list of supported architectures.

Engine guarantees:

- **Header-only and deterministic.** A verdict reads GGUF headers only, never weights, and is a
  function of the header bytes, backend, and hardware, so it is cached by header digest.
- **Resident memory.** Memory is reported as stored after import, including widened types, not as
  file size, together with what the load's worker holds in system RAM (its host tables, graph plans,
  runtime, vocabulary and prepared programs; see engine memory).
- **Catalog targets are never `Unsupported`.** That remains a release defect.
- **No slow fallback.** A family without kernels is refused with a reason, never run on a fallback.

Shared fixtures of real GGUF headers with expected verdicts (`inference/engine/fixtures/assessment`)
are tested on both sides; a change that adds support flips its fixture's expected verdict in the
same change.
