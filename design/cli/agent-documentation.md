---
applies_to:
  - cli/src/index.ts
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

The installed Magnitude skill is a small, stable entrypoint into this directory. It documents
headless operation and directs first-time onboarding to the desktop application. There is no
parallel CLI onboarding procedure or built-in Magnitude harness. Agents may perform explicit
model and connection operations through the documented commands; documentation does not create
another interactive product flow.

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
