---
applies_to:
  - packages/acn/src/local-model-rank*.ts
  - packages/acn-protocol/src/schemas/model-state.ts
  - packages/client-common/src/local-models/options.ts
  - packages/client-common/src/local-models/setup*.ts
  - cli/src/commands/inference-runtime.ts
  - packages/client-common/src/desktop/service.ts
  - desktop/src/renderer.tsx
  - web/src/components/model-center.tsx
---

# Local model ranking

Local model ranking separates stable server facts from connection-scoped user preference. While
projecting catalog state, ACN derives `LocalModelRankingScores` from each completed `Fits`
assessment with the pure ranking policy. Client-common applies the user's Fast-to-Smart preference
and memory budget to those scores. ACN does not maintain a separate ranking projection or publish
server-selected preference tiers, explanations, or a selected portfolio.

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
sorts by descending utility, breaks ties by canonical model ID, and then applies the caller's result limit. Installation state affects the row action, not ranking eligibility. Every installed
model remains available in My Models, including when it appears in Discover.

## Preference lifetime and rendering

The desktop preference is connection-scoped state and defaults to Balanced. It survives page
navigation and renderer component remounts within the connection but is not persisted. Discover
observes canonical catalog and hardware data and applies the shared client-common ranking policy.
Selectable recommendations require authoritative hardware and assessment evidence.

The five semantic positions are Fastest, Faster, Balanced, Smarter, and Smartest, corresponding to
normalized weights `0.05`, `0.25`, `0.5`, `0.75`, and `0.95`. The softened endpoints keep both speed
and intelligence relevant at every position. Discover renders these positions as a Fast-to-Smart slider and orders the
eligible catalog entries by the selected preference. Its featured set takes the first configuration
for each distinct canonical model base, then the first three models in that order. The complete
configuration ranking remains available in the catalog. Other catalog entries retain their assessment
status and cannot be mistaken for compatible recommendations. There is no memory control.

The headless CLI accepts `catalog recommendations --preference` and `--limit`. It uses the same
shared ranking policy and authoritative data as desktop Discover, with Balanced and ten results
as its defaults. It prints a finite result and does not render a chooser or keyboard controls.

## Conformance

- ACN publishes `LocalModelRankingScores`, never server-selected preference tiers or explanations.
- Scores belong to one exact catalog model configuration with a terminal `Fits` assessment.
- A model with distinct desired and effective installed configurations may temporarily have scores
  for both; the local product row uses the scores matching the configuration it currently exposes.
- Normalized score fields are named `intelligence`, `speed`, and `fidelity`; `quality` is not a
  ranking dimension.
- Missing required speed evidence fails ranking.
- Fast-to-Smart preference and the physical-memory hard filter are client-common concerns.
- Filtering and sorting happen before the requested result limit.
- Equal utility is ordered by canonical model ID only.
- Eligible installed choices are ranked by the same controls as downloadable choices.
- Every installed choice remains available in My Models, including ranked choices.
- Live native load admission remains authoritative after assessment and ranking.
