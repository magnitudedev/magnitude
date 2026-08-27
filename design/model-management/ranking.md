---
applies_to:
  - packages/acn/src/local-model-rank*.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/local-models/options.ts
  - packages/client-common/src/local-models/setup*.ts
  - cli/src/features/model-setup/chooser.tsx
  - web/src/components/local-model-onboarding.tsx
  - web/src/components/model-center.tsx
---

# Local model ranking

Local model ranking separates stable server facts from connection-scoped user preference. ACN's
`LocalModelRanker` derives `LocalModelRankingScores` for active catalog configurations with a
completed `Fits` assessment. Client-common applies the user's Fast-to-Smart preference and memory
budget to those scores. ACN does not publish server-selected preference tiers, explanations, or a
selected portfolio.

## Ranking scores

Every rankable configuration has three normalized scores in `[0, 1]`:

- intelligence is the catalog intelligence score divided by 100 and clamped to the normalized
  range;
- fidelity is the catalog fidelity rank divided by 100; and
- speed is the normalized expected generation speed at
  `min(50,000, configured context)`.

For expected tokens per second `v`:

```text
bounded = clamp(v, 0, 100)
r(bounded) = bounded / 40          when bounded <= 40
             1 + ln(bounded / 40)  when bounded > 40
speed = r(bounded) / (1 + ln(100 / 40))
```

The comparison sample is required. Missing or malformed score inputs fail ranking instead of
publishing a successful empty result. Scores are attached only to the exact assessed configuration
for which they were derived.

## Client preference

Let `p` be Fast-to-Smart clamped to `[0, 1]`, with zero meaning Fastest and one meaning Smartest:

```text
utility = intelligence ^ (0.9 * p)
        * speed        ^ (0.9 * (1 - p))
        * fidelity     ^ 0.1
```

Fidelity always contributes. Intelligence is model-level capability on the versioned Artificial
Analysis Intelligence Index scale; fidelity is artifact-variant preservation and cannot supply or
alter intelligence provenance. Memory is a hard filter and never a utility factor. A candidate is
eligible only when its assessed `memory.totalRequiredBytes` does not exceed the machine's normalized
physical-memory capacity.

The physical-memory maximum is the sum of the normalized, distinct hardware memory domains. The
system-memory total is not added separately. The client filters every local model option with scores,
sorts by descending utility, breaks ties by canonical model ID, and then returns at most ten rows.
Installation state affects the row action, not ranking eligibility. Every installed model also
remains available under `ON THIS COMPUTER`, including when it appears in the ranked group.

## Preference lifetime and rendering

Fast-to-Smart is connection-scoped setup state and defaults to `0.5`. The setup service clamps updates
to `[0, 1]`; the preference survives renderer remounts for the current connection but is not persisted.

CLI and web render the same controls and recompute results from current hardware, options, and
control state. Selectable ranked models require an authoritative hardware result. Fast-to-Smart has
five visually equidistant semantic positions: Fastest, Faster, Balanced, Smarter, and Smartest,
corresponding to normalized weights `0.05`, `0.25`, `0.5`, `0.75`, and `0.95`. The softened endpoint
weights ensure both speed and intelligence remain ranking factors at every position. The CLI
calculates tick spacing from label widths so labels cannot overlap. Its track, unselected ticks, and
unselected labels use the normal white text color; only the selected tick and its label use accent
blue, with no separate marker glyph.
The CLI renders the Left/Right preference instruction in the chooser's shared bottom control row
alongside model navigation, selection, and exit controls. The scale is not focusable or
pointer-selectable. CLI keyboard traversal contains only model rows;
Left/Right and `h`/`l` adjust Fast-to-Smart regardless of the selected row. The cursor owns a visible
row position, so re-ranking keeps it on the same rank while the model and details at that rank change.
There is no memory control.

## Conformance

- ACN publishes `LocalModelRankingScores`, never server-selected preference tiers or explanations.
- Scores belong to one exact catalog model configuration with a terminal `Fits` assessment.
- Normalized score fields are named `intelligence`, `speed`, and `fidelity`; `quality` is not a
  ranking dimension.
- Missing required speed evidence fails ranking.
- Fast-to-Smart preference and the physical-memory hard filter are client-common concerns.
- Filtering and sorting happen before the ten-result limit.
- Equal utility is ordered by canonical model ID only.
- Eligible installed choices are ranked by the same controls as downloadable choices.
- Every installed choice remains available under `ON THIS COMPUTER`, including ranked choices.
- Live native load admission remains authoritative after assessment and ranking.
