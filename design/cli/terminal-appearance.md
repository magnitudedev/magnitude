---
applies_to:
  - cli/src/index.tsx
  - cli/src/hooks/use-theme.tsx
  - cli/src/platform/terminal-appearance.ts
  - cli/src/types/theme-system.ts
  - cli/src/utils/theme.ts
  - cli/src/**/*.tsx
  - cli/src/**/*.ts
---

# Terminal appearance

## Purpose

The terminal is the authority for the CLI's ambient background and light-or-dark appearance.
Magnitude derives one coherent theme from those facts without requiring a user-selected light or
dark mode. Magnitude's own palette remains the authority for application colors.

## Appearance lifecycle

Terminal appearance consists of the terminal's reported default background and light-or-dark
classification. It is resolved before the first application render so bootstrap and failure screens
cannot flash an incompatible fallback theme.

The detected default background is the primary evidence for light-or-dark classification. The
renderer-reported mode is secondary evidence, followed by standard environment hints. Missing or
malformed evidence falls back safely and never prevents startup.

The renderer owns live appearance observation. Theme-mode and palette events, terminal color-scheme
notifications handled by the renderer, and focus-based recovery for terminals without update support
may trigger a palette refresh. Refreshes are coalesced. A failed palette refresh preserves the last
known appearance unless the renderer independently reports a new mode, in which case the matching
mode fallback replaces the stale background. Renderer shutdown releases all listeners and pending work.

## Theme contract

The CLI exposes one resolved theme. Its vocabulary describes reusable visual meaning: text
hierarchy, backgrounds, borders, accents, activity, status, plan and shell accents, diff
backgrounds, Markdown, and syntax. It does not describe individual screens or components.

Components consume resolved theme values only. They do not:

- inspect terminal appearance or light-or-dark mode;
- choose raw palette shades;
- define light and dark alternatives;
- infer terminal color capability; or
- manufacture feature-specific theme extensions.

Hover, focus, and selection share the general accent unless their content has an existing semantic
role. Agent work, activity-rail work, and autopilot activity share one activity pulse. Radar visuals
use the general accent, border, and text hierarchy. Shell output uses ordinary text and status
colors, with only the shell mode accent remaining distinct.

## Color derivation

The terminal background selects the fixed dark or light Magnitude mapping. Magnitude's palette
remains authoritative for its neutral, brand, and semantic colors. The resolver may adjust their
lightness only by selecting an explicit light-mode shade from the same Magnitude scale; it does not
mix or recolor palette values at runtime. It must not substitute user-configured ANSI hues for
Magnitude blue, violet, green, orange, red, or slate. ANSI palette entries are terminal-profile
choices, not stable semantic colors. The dark mapping preserves the established CLI palette exactly.

The canvas is transparent, and the detected terminal background is retained for UI that must cover
the ambient terminal exactly. Other opaque backgrounds come from the fixed dark or light Magnitude
mapping. Contrast tests cover the text hierarchy and essential semantic colors against representative
dark, light, and warm-light terminal backgrounds.

OpenTUI capabilities are authoritative for truecolor and indexed-color behavior. Emulator-name
heuristics are not part of theme resolution.

## Conformance

- The first rendered frame is readable against the detected terminal background.
- Runtime terminal-theme changes update the resolved theme without restarting the CLI.
- No user appearance preference is required or persisted.
- Feature code contains no raw color-scale selection or light/dark branching.
- Failed detection or refresh retains a mode-consistent fallback or the last known good appearance.
- Text and essential indicators meet their declared contrast contracts.
