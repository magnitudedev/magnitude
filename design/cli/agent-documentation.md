---
applies_to:
  - cli/src/index.tsx
  - cli/src/commands/docs.ts
  - cli/src/agent-docs/**
  - cli/src/types/text-imports.d.ts
  - packages/roles/src/prompts/leader.txt
---

# Agent documentation directory

Magnitude ships a small directory of product documentation for language models operating through
the agent's shell. Each topic has a stable identifier, a short description, and Markdown content.

`magnitude docs` prints the complete topic directory. `magnitude docs <topic-id>` prints the exact
Markdown for one topic. Both operations are local to the CLI process: they do not initialize the
interactive client, connect to ACN, read user state, or use the network.

The topic corpus is distinct from the public documentation site and from internal engineering
documents. Its Markdown is bundled into the compiled CLI executable. The leader prompt advertises
the lookup mechanism without adding topic contents or the topic list to every context.

The installed Magnitude skill is a small, stable entrypoint into this directory. It routes
agent-guided setup to the `onboarding` topic, which owns the current CLI sequence, interpretation of
machine-specific catalog evidence, user model-selection conversation, progress observation, and
harness connection behavior. Keeping the procedure in the CLI documentation corpus lets a refreshed
CLI provide current guidance without requiring an already-installed skill copy to duplicate it.

The onboarding topic begins with an explicit conversational contract: agents briefly explain major
actions, narrate meaningful progress, communicate choices and tradeoffs clearly, and welcome user
questions throughout the process. It directs Magnitude-specific questions to the bundled topic
directory first. It distinguishes service registration from first startup, model
acquisition from model loading, and persistent harness selection from the already-running session.
It gives agents adaptive polling guidance that balances estimated operation duration with meaningful
user updates, waits for the catalog's authoritative discovery and assessment completion states,
starts model selection with balanced recommendations, uses Faster and Smarter for ordinary
directional requests, and reserves Fastest and Smartest for clearly requested extremes after
explaining what the opposing signal gives up. It names every supported harness ID and its
harness-specific handoff. Magnitude selection is automatic, Pi exposes the supported
current-process switching path, and every other external harness requires exiting the running
process with Ctrl+C and relaunching from the printed command. Persistent harness model choice is
communicated as selecting a model, not setting a default. Guidance defines outcomes and decision
boundaries without scripting exact user-facing sentences. The procedure must not claim completion
signals or side effects that the non-interactive CLI does not expose.

The `speculative-methods` topic owns the self-contained user-facing explanation of the acceleration
methods reported by Magnitude. It defines their practical typical ordering, explains the mechanism
behind each method, and states that Magnitude acquires, validates, and activates reviewed draft
material automatically. It must distinguish a rule of thumb from machine-specific speed evidence
and must not present an acceleration method as model intelligence or quality.

The `recommendations` topic owns the user-facing explanation of recommendation eligibility,
preference tradeoffs, and displayed evidence. It explains the durable, practical meaning of speed,
memory, context, intelligence, artifact accuracy, acceleration, capabilities, and canonical
identity without exposing ranking formulas or fixed assessment parameters. It identifies the
Artificial Analysis Intelligence Index and gives dated, methodology-qualified frontier scores as
familiar reference points for explaining a local model's score. It cross-references
`speculative-methods` rather than duplicating that method guide. It presents Faster and Smarter as
the normal directional preferences and Fastest and Smartest only as explicit extremes.

## Conformance

- Topic lookup is exact and case-sensitive over one flat namespace.
- Directory output is deterministic and contains every registered topic.
- Successful topic output is raw Markdown on stdout with one trailing newline.
- Unknown topics produce a useful stderr diagnostic and a nonzero exit status.
- Documentation lookup works without ACN, network access, or a source checkout.
- Only explicitly registered Markdown is published as agent documentation.
