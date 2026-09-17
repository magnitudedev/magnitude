---
applies_to:
  - cli/src/commands/output.ts
  - cli/src/commands/output.test.ts
  - cli/src/index.ts
---

# Headless terminal output

The CLI renders ordinary human-readable text, tables, and labeled fields. It does not probe terminal
appearance, construct a theme, render a TUI, or watch terminal color changes. Redirected output
contains no cursor control or animation. Narrow terminals retain complete addressable identifiers.

Desktop appearance uses the canonical shared Magnitude palette and the existing light, dark, and
system preference, as specified in `design/clients/desktop-inference.md`.
