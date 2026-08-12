---
applies_to:
  - packages/client-common/src/hooks/use-slash-commands.ts
  - packages/client-common/src/commands/**
  - cli/src/features/composer/**
  - web/src/components/composer*
---

# Composer command selection

The slash menu distinguishes commands that act immediately from selections that prepare a message.

- Built-in commands execute when selected.
- Skills are message context. Selecting one replaces the slash query with `/<skill> ` and leaves the
  draft focused for the user to complete.
- Selecting a skill never sends a message, creates a session, changes message history, or removes
  attachments. Only an explicit composer submission sends the resulting skill-prefixed message.
- Draft population closes the slash menu and places the cursor after the trailing space.
- Explicit submission routes recognized, argumentless built-in commands even after whitespace closes
  the menu. Skills and argument-bearing slash text remain ordinary messages.

## Acceptance criteria

1. Keyboard and supported pointer selection have identical semantics.
2. CLI and web use the same command-selection classification.
3. Built-in command behavior is unchanged.
4. Skill selection and message submission are distinct user actions.
5. A handled built-in submission does not send a message or enter message history.
