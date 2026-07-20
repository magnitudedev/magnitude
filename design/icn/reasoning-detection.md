---
applies_to:
  - inference/crates/icn-reasoning/**
  - inference/crates/icn-contracts/**
  - inference/crates/icn-models/**
  - inference/crates/icn-api/**
  - inference/crates/icn-engine/**
  - inference/native/llama-cpp-rs/llama-cpp-2/src/common_chat.rs
  - inference/native/llama-cpp-rs/llama-cpp-sys-2/wrapper_common_chat.cpp
  - inference/native/llama-cpp-rs/llama-cpp-sys-2/wrapper_common_chat.h
  - inference/native/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/common/chat.cpp
  - inference/native/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/common/chat.h
---

# ICN reasoning detection

ICN discovers how the effective chat template controls reasoning and presents that behavior as an
ordered list of normalized reasoning-effort options. Each option has a model-specific rendering
recipe retained inside ICN. Callers see stable normalized names rather than model-specific Jinja
arguments.

This design governs the Rust inference implementation. There is no Bun reasoning-inspection
implementation or fallback; ACN and clients consume the completed ICN result.

## Normalized behavior

`none` is a first-class normalized option meaning reasoning disabled. ICN includes it whenever the
effective template has a verified way to disable reasoning.

A model with a simple thinking toggle is presented as `none` and `high`. In that case, `high` means
the enabled side of the toggle; it does not claim that the model has a native high-effort setting.
The distinction remains internal to ICN so the caller can use the same option vocabulary across
models.

The ordinary normalized ordering is:

| Order | Normalized option | Meaning |
| --- | --- | --- |
| 1 | `none` | Reasoning disabled |
| 2 | `minimal` | Lowest declared enabled effort |
| 3 | `low` | Low declared effort |
| 4 | `medium` | Medium declared effort |
| 5 | `high` | High declared effort, or enabled for a toggle-only model |
| 6 | `xhigh` | Extra-high declared effort |
| 7 | `max` | A distinct maximum setting where the model defines one |

Native `off`, `no_think`, and `disabled` spellings normalize to `none`. Native `extra_high`,
`extra-high`, `very_high`, and `xhigh` spellings normalize to `xhigh`. Aliases never appear as
duplicate public options. `max` is not collapsed into `xhigh`, because current GLM and DeepSeek
templates distinguish high from max.

Some templates expose a meaningful named state outside this scale. MiniMax M3's adaptive mode is
the important current example. Such a state remains `adaptive`; it is not mislabeled as medium.

The normalized default is the behavior produced by pinned llama.cpp. Its common-chat input exposes
`enable_thinking` as a boolean whose default is enabled, so a caller that does not choose an effort
uses the enabled side of a supported toggle. ICN does not maintain a second parser for a template's
authored Jinja fallback.

## Model-template formats

Reasoning controls in current templates fall into several recurring formats. Detection is based on
the effective template behavior, not the model filename. Family names below are concrete examples
and test requirements, not runtime dispatch keys.

### Boolean thinking control

Qwen 3.5 and Qwen 3.6 use `enable_thinking`. Kimi K2.5 and Kimi K2.6 use a similar boolean named
`thinking`. Gemma 4 also uses `enable_thinking`; all are rendered through the pinned common-chat
input contract.

| Example | Native behavior | Normalized options | Normalized default |
| --- | --- | --- | --- |
| Qwen 3.5 | `enable_thinking` on or off | `none`, `high` | `high` |
| Qwen 3.6 | `enable_thinking` on or off | `none`, `high` | `high` |
| Kimi K2.5 | `thinking` on or off | `none`, `high` | `high` |
| Kimi K2.6 | `thinking` on or off | `none`, `high` | `high` |
| Gemma 4 | `enable_thinking` on or off | `none`, `high` | `high` |

The private recipe for `high` uses the template's actual boolean key. ICN never assumes that
normalized `high` should be passed as a native `reasoning_effort` string.

Qwen 3.6, Kimi K2.6, and Gemma 4 also control whether reasoning from earlier assistant messages is
retained. That history behavior must be detected separately from generation-time reasoning. It
does not create additional effort options.

### Fixed reasoning

Some templates always produce or preserve reasoning and expose no caller control. Kimi K2.7 Code
and the MiniMax M2 family are representative.

| Example | Native behavior | Normalized options | Normalized default |
| --- | --- | --- | --- |
| Kimi K2.7 Code | Thinking is fixed on | `high` | `high` |
| MiniMax M2/M2.5/M2.7 | Thinking is fixed on | `high` | `high` |

`none` is not advertised for these templates. A request for `none` is rejected rather than silently
running the model with reasoning enabled.

### Toggle plus discrete symbolic effort

GLM-5.2 supports disabling thinking and distinguishes high from max. DeepSeek V4 exposes the same
product choices through a different combination of native mode and effort controls.

| Example | Native format | Normalized options | Normalized default |
| --- | --- | --- | --- |
| GLM-5.2 | Boolean thinking plus high/max effort | `none`, `high`, `max` | `max` |
| DeepSeek V4 Flash | Chat/thinking mode plus high/max effort | `none`, `high`, `max` | `high` |
| DeepSeek V4 Pro | Chat/thinking mode plus high/max effort | `none`, `high`, `max` | model/template default |

For GLM-5.2, the official template treats exactly `high` as high and routes the omitted or other
branch to max. Invalid-value probing alone cannot prove the intended name `max`; known-template
semantic evidence is required.

For DeepSeek V4, disabling may require `thinking_mode="chat"`, while enabled options require the
thinking mode plus the correct native effort. The published repositories may use a custom encoder
rather than embedding the same Jinja template later found in a GGUF conversion. ICN therefore
classifies the effective local template, not the upstream repository name.

### String-valued reasoning mode

DeepSeek V3.2 uses chat/thinking mode values. MiniMax M3 uses disabled/adaptive/enabled values.

| Example | Native modes | Normalized options | Normalized default |
| --- | --- | --- | --- |
| DeepSeek V3.2 | `chat`, `thinking` | `none`, `high` | template default |
| MiniMax M3 | `disabled`, `adaptive`, `enabled` | `none`, `adaptive`, `high` | `adaptive` |

ICN preserves adaptive as a separate choice. It does not infer an ordering between adaptive and
high from prompt differences.

### Open pass-through symbolic effort

GPT-OSS interpolates a reasoning-effort value into the prompt. Arbitrary strings can therefore
change rendering even though the documented model domain is low, medium, and high.

The normalized GPT-OSS profile is `low`, `medium`, and `high`, with medium as the default. Those
semantics come from versioned model/template evidence and are verified against the effective
template. ICN does not expose random strings merely because the Jinja program accepts them.

If an unknown template passes through arbitrary strings and has no trusted declaration, ICN does
not invent an effort domain.

### Native prompt budget

Seed OSS uses a numeric `thinking_budget`: zero disables reasoning, negative one requests an
unbounded mode, and positive values request a prompt-level budget. Inkling accepts named or numeric
effort values over a declared range.

These formats demonstrate why template effort, native prompt budgets, and hard inference budgets
are different concepts. In the initial implementation, ICN may recognize these controls as
template facts, but it does not synthesize a public numeric effort range or automatically activate
a token budget.

### History controls

Current families use several independent history conventions:

| Family examples | Native history control |
| --- | --- |
| Qwen 3.6, Kimi K2.6, Gemma 4 | Preserve previous thinking |
| GLM-5/5.1/5.2 | Clear previous thinking |
| DeepSeek V3.2/V4 | Drop previous thinking |
| Kimi K2.7 Code | Fixed preservation behavior |

Detection must include prior assistant reasoning and tool-result histories so these controls are not
mistaken for ignored arguments. History behavior is retained as reasoning metadata; it does not
change the normalized effort list in this implementation.

## Detection architecture

Detection loads the actual model through llama.cpp in no-allocation mode and constructs common-chat
templates from that model. This makes llama.cpp authoritative for metadata, vocabulary, template
selection, BOS/EOS behavior, and rendering without loading tensor weights or creating an inference
context. ICN does not copy those inputs into a parallel metadata representation.

The process has two distinct responsibilities:

1. Template inspection establishes observable rendering facts.
2. Normalization combines those facts with trusted semantic evidence and produces the public option
   list plus private model-specific recipes.

Keeping these responsibilities separate prevents a prompt difference from being treated as proof
of model semantics.

### Probe coverage

ICN renders omitted, enabled, and disabled variants of known boolean controls; known string modes;
canonical effort values and their aliases; native prompt-budget sentinels; and reasoning-history
controls. It also renders two randomized invalid effort strings.

Every control is tested across multiple conversations:

- a plain user generation;
- system plus user messages;
- tools supplied before a tool call;
- an assistant tool call followed by a tool result;
- prior assistant reasoning followed by another user turn;
- reasoning interleaved with tool calls and results.

Each conversation contains a nonce that must survive rendering. A failed conversation shape is
recorded independently; it does not erase successful evidence from other shapes.

Comparison includes the complete prepared behavior that affects inference: prompt text, generation
prompt, parser selection, reasoning markers, grammar, preserved tokens, and added stop sequences.

### What probing establishes

Differential rendering can establish that:

- a boolean or string mode changes behavior;
- two native values are aliases;
- a value is equivalent to the omitted default;
- invalid values are rejected, ignored, or share a fallback;
- arbitrary values are passed through;
- an effort only matters when thinking is enabled;
- a control affects reasoning history rather than initial generation.

Probing does not establish that one prompt is higher quality, that changed strings form an ordered
scale, or that a pass-through value was used during training.

One meaningful alternate effort is retained. The inspector does not require two non-default values
before recognizing a real control. Default detection keeps baseline-equivalent values available;
it does not discard them before asking which value matches omission.

### Semantic evidence

Known semantics that cannot be proven from rendering are held in a small, versioned registry tied
to exact effective-template fingerprints. Examples include GPT-OSS's documented low/medium/high
domain and GLM-5.2's intended max fallback.

Repository identity may support a conclusion but cannot override contradictory effective-template
behavior. A Qwen-named GGUF carrying a modified fixed-thinking template is classified as fixed
thinking. Runtime behavior is never selected from filename substrings.

Changing the effective template, template-selection inputs, inspector version, semantic-policy
version, or pinned llama.cpp behavior invalidates the cached result.

Reasoning persistence is part of the artifact-inspection index in the
[model-management cache](./model-management.md#model-derived-cache). The reasoning detector defines
the typed result and a stable semantic-policy identity included in the inspection evidence; it does
not own a reasoning-specific cache or filesystem layout. Local inventory inspection and remote
preview reuse the same index whenever the artifact, effective template, and policy evidence match.

## Request behavior

The Rust API accepts one normalized `reasoning_effort`. Omitting it selects the normalized default.
After the target model and effective template are known, ICN validates that the requested option is
present and applies its private native recipe.

Examples:

| Request | Effective template | Native behavior applied by ICN |
| --- | --- | --- |
| `none` | Qwen 3.6 | Disable `enable_thinking` |
| `high` | Qwen 3.6 | Enable `enable_thinking` |
| `none` | Kimi K2.6 | Disable `thinking` |
| `high` | Kimi K2.6 | Enable `thinking` |
| `none` | DeepSeek V4 | Select chat mode |
| `high` | DeepSeek V4 | Select thinking mode and native high effort |
| `max` | GLM-5.2 | Enable thinking and select native max behavior |
| `none` | Kimi K2.7 Code | Reject as unsupported |

Normalized strings are never blindly forwarded as Jinja values. A recipe is valid only for the
same effective-template fingerprint used during detection. A template change requires resolution
against the new profile.

An explicit unsupported option fails with the model's supported option list. ICN does not silently
fall back to the default. Raw template arguments remain an advanced escape hatch when normalized
reasoning is absent. A request that supplies both normalized reasoning and conflicting raw
reasoning controls is rejected.

## Token budgeting

The inference layer retains a place for an optional automatic hard reasoning budget on each
normalized option. This allows a future policy to associate, for example, low or high with a token
limit without changing detection or request routing.

Automatic budgeting is disabled for every option in the initial implementation. Selecting minimal,
low, medium, high, xhigh, or max does not install the Bun implementation's 1K, 2K, 4K, or 8K
heuristic.

Caller-supplied `thinking_budget_tokens` remains an independent explicit inference control. Native
prompt controls such as Seed's `thinking_budget` are also distinct from a hard decoder-enforced
limit. A future policy must preserve that distinction and may enable a hard automatic budget only
when the selected template behavior has reliable reasoning boundaries.

## Failure semantics

For a valid, stable, runtime-supported template, reasoning detection produces a complete result.
A successful no-reasoning result normalizes to the single option `none`.

Template compilation failure, inability to render the required baseline, unstable rendering, lost
probe nonces, or inability to derive the declared public property is an assessment failure. It is
not normalized to `none` and is not cached as a model capability.

Fixed reasoning is also a successful result. It normalizes to `high`, not `none`, and does not
advertise a disabling option.

## Acceptance criteria

- Qwen 3.5/3.6 boolean controls normalize to `none` and `high` with the pinned common-chat default.
- Kimi K2.5/K2.6 nonstandard booleans normalize to `none` and `high`.
- Kimi K2.7 Code and MiniMax M2 fixed reasoning normalize to `high` only.
- Gemma 4 normalizes to `none` and `high` with the pinned common-chat default.
- GLM-5.2 preserves distinct `none`, `high`, and `max` options.
- DeepSeek V3/V4 modes map to the correct normalized options and private mode recipes.
- MiniMax M3 preserves `adaptive` rather than relabeling it as an effort.
- GPT-OSS exposes only its declared low, medium, and high values despite open pass-through Jinja.
- Native aliases collapse to one normalized option in deterministic order.
- One meaningful alternate effort is not discarded.
- No fixed-thinking template is advertised as disableable.
- Explicit unsupported requests fail rather than falling back.
- Every private recipe is bound to the effective-template fingerprint.
- Every normalized option has automatic token budgeting disabled initially.
- Selecting a symbolic effort never implicitly sets a hard reasoning-token budget.
- Detection is model-free, bounded, deterministic apart from nonces, and cacheable.
