---
applies_to:
  - cli/src/commands/**
  - cli/src/index.ts
  - cli/src/server/application.ts
  - cli/src/server/server-setup*.ts
  - cli/src/server/local-harness-connections.ts
  - cli/src/agent-docs/**
  - packages/harness-connections/**
  - packages/sdk/**
---

# Non-interactive CLI contract

## Scope

The non-interactive CLI is a human-readable, agent-usable projection of Magnitude product state.
It does not expose transport documents or internal state graphs. Bare `magnitude` prints help and
exits. Help, version, and documentation do not start the desktop or service. There is no terminal
onboarding, chat harness, or hidden hosted-setup mode. All onboarding lives in the desktop
application. `server setup` registers Magnitude as a system service; it is not onboarding.

The public command vocabulary is:

```text
update [check | status | download | install | discard]
app open
serve
server setup | remove
status
hardware
catalog status | list | show <model-id> | recommendations [--preference <value>] [--limit <count>]
catalog search <query> [--limit <count>]
catalog inspect <huggingface-url | owner/repo | owner/repo:QUANT>
catalog pull <model-id | owner/repo[:QUANT]> [--drafter auto|none|<ref>] [--force] [--only projector|drafter]
catalog cancel <model-id> | remove <model-id>
models status [model-id] [--all] | load <model-id> | stop [model-id]
models drafter <model-id> <ref | auto | none>
models hide <model-id> | unhide <model-id>
models folders list | add <path> | remove <path>
models folders enable <location> | disable <location>
connections list | connect <harness> [--set-model <model-id>] [--install-skill]
connections sync [harness] | remove <harness>
hf login | status | logout
docs [topic-id]
```

Every inference command (`hardware`, `catalog`, `models`) and every `hf` command accepts `--json`. Nothing prompts:
ambiguity resolves the way the desktop's defaults do, and the choice is printed. Removed output flags are not retained as aliases.
The acquisition command remains `catalog pull`. Plugins call the SDK over RPC instead of the CLI.

## Domain ownership

- `serve` owns the foreground application and service tree until shutdown or cooperative Desktop handoff.
- `server setup` and `server remove` register and remove the system service on Linux (systemd) and
  macOS (launchd), running as the person and asking for authorization only for their one root step.
  On Windows they explain that server mode isn't available and change nothing.
- `status` passively reports owner and runtime readiness; tray and login-startup fields appear only for Desktop. With no owner, it prints startup guidance (for the system service where one is set up) and exits successfully without starting anything.
- `hardware` reports the local inference topology, current memory use, and current allocation.
- `catalog` finds and downloads models: catalog assessment progress, reviewed model choices,
  machine-specific assessment evidence, recommendations, Hugging Face search and inspection, and
  download operations.
- `models` covers what is on this computer: catalog and custom models present or undergoing local
  operations, imports, drafters, hidden models, model locations, and runtime residency.
- `connections` reports harness installation and observed configuration integrity, and writes
  harness configuration for the person running the command, from the CLI process.
- `hf` signs the service in to and out of Hugging Face and reports the account it uses.

Catalog output never includes acquisition or residency state, except that `catalog inspect` marks a
version already on this computer. Model-status output never includes
catalog ranking or provenance. The focused `models status <model-id>` view is the observation point
for download and load progress.

## Presentation

Comparable collections use borderless tables. Heterogeneous details use labeled fields. Mutations
use concise acknowledgements and include an exact observation command when work continues in the
background. A table that does not fit the terminal becomes labeled row blocks; canonical model and
harness IDs are never truncated.

Output uses friendly names as the primary identity and prints the exact canonical ID whenever an
object can be addressed by another command. Empty state is a successful, explicit sentence. Normal
errors write one actionable product message to stderr and exit nonzero. Redirected output contains
no cursor control, animation, or ANSI dependency.

Normal output excludes ACN/ICN terminology, assessment and environment IDs, package identities,
cache paths, native device indices, source revisions, raw tags, ranking utility, operation IDs,
retryability flags, and stack traces.

`--json` prints one machine-readable document of the same product facts the text form shows,
including canonical IDs, structured unsupported reasons with their developer detail, and drafter
verdicts. It never includes internal ACN/ICN identities, package structure, or prose meant only for
terminals: no assessment or environment IDs, memory-domain, accelerator, or device identities
(per-domain memory evidence and the domain-named limiting resource are omitted, and totals remain),
managed installation paths or file ownership (installed size remains; a custom model's own file is
its origin), or raw ranking inputs (ranking scores and fidelity rank). Catalog documents (`catalog
list`, `show`, and `recommendations`) carry no acquisition or residency state. Each document is
encoded through an explicit CLI JSON schema, so a protocol field reaches `--json` only when that
schema names it. Errors stay on stderr with a nonzero exit. Under `--json` every inference command
writes its error to stderr as one document, `{"error": {"message": <product message>, "code":
<stable snake_case code>}}`, so stdout holds only a successful result. The code is the service's
failure code when it has one, else the failure's kind. One shared command runner produces this
document for every command, and it reports an unexpected defect, such as a response that fails to
decode, the same way instead of crashing.

Memory uses hardware-conventional units; storage and transfer use decimal units; context uses
compact token counts; generation speed uses `tok/s`. Rounded values are presentation only.

## Client connections

The CLI provides human-oriented commands. Plugins use the private bundled Effect SDK and the
existing RPC endpoint for model observation and control. Both connect to an already-running
Desktop or Headless owner without launching an application. `connections connect` and `sync` need
the running service's models, which they read over loopback, and write configuration themselves. With no owner, service-backed CLI
commands fail with “No Magnitude service is running. Open the Magnitude desktop app or run
`magnitude serve`.” Passive `status` reports absence successfully. `--json` output is a product
projection for scripts and agents, not an SDK substitute; there is no CLI service starter or
separately published integration-contract package.

## Catalog and recommendation behavior

`catalog status` reports authoritative assessment completion and progress counts when targets are
available. When assessment is incomplete and no targets have been reported, it says so without
inferring a preparation phase or failure reason from the empty count. It does not infer
completion from catalog rows or recommendation availability and does not wait for assessment to
finish.

`catalog list` displays only assessed catalog configurations that fit the current machine. It shows
friendly identity, predicted memory, baseline speed, configured context, speculative acceleration,
and canonical ID. Outstanding or failed assessments are summarized after the useful rows.

`catalog recommendations` reuses the shared onboarding eligibility and ranking policy. Preference
is one of Fastest, Faster, Balanced, Smarter, or Smartest and defaults to Balanced; limit defaults
to ten. Recommendation evidence includes speed, memory, context, intelligence, artifact fidelity,
acceleration, capabilities, and canonical ID. Raw ranking scores and aggregate utility remain
private. The command is a client projection over existing catalog and hardware authorities, not a
second server recommendation authority. Successful nonempty output ends with the exact local
documentation command for interpreting the evidence and ranking methodology.

`catalog show` supplies one model's useful curated and machine-specific evidence and prints the
`catalog search` command that finds other versions of it. Successful `catalog recommendations`
output ends with a hint line for adding a model from Hugging Face.

`catalog search` lists Hugging Face repositories with uploader, the catalog model each is a variant
of, and downloads; results of architectures Magnitude doesn't run follow in their own table with
that reason. `catalog inspect` prints the repository step: versions with sizes and the engine's
verdict, the recommended version, and drafter choices in
[drafter selection](../model-management/drafter-selection.md) order with their labels. It accepts a
Hugging Face URL and makes no changes.

No catalog or model observation command waits for assessment or residency to settle. Service health
is likewise independent of background assessment.

## Model operations

`catalog pull` converges a catalog model to installed and current. Given a Hugging Face repository,
it adds that model pinned to the repository's current commit: with no quant it picks the
recommended version, with no `--drafter` it uses `auto` (the preselected drafter), and it prints
what it chose and the model ID to load. A version already on this computer downloads only the
drafter, and says so. A version the engine can't run is refused with its reasons, and one that does
not fit is refused with how much more memory it needs ("needs X GB more memory than this computer
can provide") unless `--force` is given; ACN enforces both (`ModelVersionNotRunnable`), so the
desktop, which disables those versions, and the CLI agree. `--force` applies only to Hugging Face
versions and never admits an unsupported one. Pull acknowledges admitted work by what it is, from the sync outcome: a first
or resumed download, an update to a newer revision, or a missing drafter, and directs the caller to
focused model status. `--only projector|drafter` downloads just that component of a catalog model
whose exact target was found on this computer without it (`DownloadCatalogComponent`); it applies only
to catalog models and not with `--drafter` or `--force`. Pull, cancel, and removal validate only
argument syntax before delegating directly to their authoritative mutations.

`models drafter` adds,
changes, or turns off the drafter of a custom model and is refused for catalog models. `models hide`
and `unhide` apply to found models, including a catalog model whose files are only in an external
location; `catalog remove` on such a model explains where the files are and points at `models hide`. `models folders list` shows the automatic locations and the
user's folders, one row per directory: an automatic location lists each directory found or unreadable
(every directory checked when none exists), each Found, Off, Not found, or Can't read, with what to
do for each directory Magnitude can't read; `add` and `remove` manage the folders, and `enable` and
`disable` turn an automatic location (`huggingface-cache`, `lm-studio`, `llama-cpp`, or `ollama`) on
or off, which applies to all its directories.

`models status` lists catalog models on the computer or with relevant acquisition/removal work and
every visible custom model, with its source, the catalog model it is a variant of, and its drafter
(the attached drafter, "Turned off", or the recommended drafter's "Faster with a drafter · Add"),
followed by a footer with one line per catalog component an installed catalog model lacks ("Image
input for … isn't downloaded" with its `--only` pull command) and counts of models that can't run and
hidden models; `--all` lists those with their reasons and lists hidden models marked hidden. The addressed form adds Source, Variant of, Drafter,
and the drafter candidates with their labels and verdicts; for a catalog model, a line per missing
component ("Image input  Not downloaded (1.03 GB) - magnitude catalog pull <id> --only projector"), which
is never reported as an update. Addressed model commands
accept every canonical model ID, hidden models' included; the addressed form of a hidden model says
it is hidden and how to unhide it. One status field applies product priority:
removal, transfer, failures, load/stop, ready, update availability, then unloaded. An optimizing
model reports `Optimizing` with its percentage once tuning is measured. The addressed
form reports installation, transfer or optimization progress, runtime, memory, context, and
actionable failure details without historical or internal operation state.

Load acknowledges admission and prints the focused status command; it never claims readiness from
the load acknowledgement. Load and stop delegate directly to their authoritative mutations. Stop without a model
stops the active one; stop naming a model that isn't loading or running succeeds and says it wasn't running.

## Hugging Face account

There is one Hugging Face sign-in, owned by the service and shared with the desktop. The CLI never
sees, accepts, or reads a token: no pasted token, no `HF_TOKEN`. `hf login` uses the device
flow so it works over SSH and headless: it starts a device sign-in, prints the verification page,
the code, and how long the code is valid, and never opens a browser. The service polls Hugging Face;
the CLI follows its own attempt (by the attempt ID the start returned) about once a second until it
ends, so another client's sign-in or an account left signed in by an earlier attempt never answers
for it. It then prints “Signed in to Hugging Face as <username>” and exits 0, or prints the shared
reason (declined, expired, failed) and exits 1. An attempt cancelled elsewhere fails as cancelled,
and one replaced by a newer sign-in fails as superseded. Ctrl+C or a termination signal cancels
this attempt, and only it, before exiting with status 130. Under `--json`, login prints exactly two
lines on success: the device sign-in `{attemptId, verificationUri, verificationUriComplete?, userCode, expiresIn, interval}`
(seconds; `verificationUriComplete` only when Hugging Face returns it), then the account; a failure leaves only the first line on stdout
and writes the usual error document, whose code names the reason
(`hugging_face_sign_in_denied`, `_expired`, `_failed`, `_cancelled`, `_superseded`).

`hf status` prints the account: signed in as whom, a sign-in in progress, or signed out
with the reason and the login command. `hf logout` removes the service's saved sign-in.
Under `--json` both print the account document, `{"_tag": "SignedIn", "username", "avatarUrl"?}`,
`{"_tag": "SigningIn", "method"}`, or `{"_tag": "SignedOut", "reason"?}`.

A gated repository is refused with what it needs. Without a sign-in: “Sign in to Hugging Face to
use this model. Run `magnitude hf login`.” Without accepted terms: “Accept this model's
terms on Hugging Face: <repository page> — then try again. Some models also need the author's
approval.” The same wording applies to commands the service refuses with
`HuggingFaceAccessRequired` and to download failures in model status.

## Connections

Connection observation reports executable installation and configuration integrity separately;
a missing executable cannot hide intact or unreadable configuration.
`Connected` comes from actual configuration and required artifacts, not the durable receipt or executable detection. Connection
mutations delegate directly to the shared connector service, run in the CLI process for the person
running the command; the service never writes them for the CLI. Success reports configuration and
artifact installation. There is no launch-plan or handoff output and no Magnitude harness destination.
Users launch their external harness themselves.

## Conformance

- Every public command and option has useful help.
- Collection ordering is deterministic and every collection has an explicit empty state.
- Addressable rows preserve exact canonical IDs at every terminal width.
- Catalog status preserves the authoritative assessment completion state.
- Recommendations match shared onboarding ranking and memory eligibility.
- No observation command waits for assessment or residency completion.
- Mutation commands acknowledge authoritative completion without adding preflight state
  interpretations.
- Agent documentation directs onboarding to the desktop and documents headless model/configuration commands.
- Tests cover collection, detail, narrow-width, empty, partial, failure, and redirected forms.
- Every inference command supports `--json`, and no command prompts.
