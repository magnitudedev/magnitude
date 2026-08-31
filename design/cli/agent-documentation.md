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

## Conformance

- Topic lookup is exact and case-sensitive over one flat namespace.
- Directory output is deterministic and contains every registered topic.
- Successful topic output is raw Markdown on stdout with one trailing newline.
- Unknown topics produce a useful stderr diagnostic and a nonzero exit status.
- Documentation lookup works without ACN, network access, or a source checkout.
- Only explicitly registered Markdown is published as agent documentation.
