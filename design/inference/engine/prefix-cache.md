---
applies_to:
  - inference/engine/scheduler/**
  - inference/engine/executor/src/domain/**
  - inference/engine/generation/**
---

# Prefix cache

**A request resumes from the deepest cached state on its path, whatever sequence of requests
came before.** One abstraction owns prefix reuse: the prefix cache. Nothing else decides whether
cached state is valid for a request, and nothing re-checks that decision.

## Concepts

- **Path**: the input a numerical state has consumed: a token sequence plus the media identities
  conditioned on it, ending at an exact layout boundary. A request's path is its prompt followed
  by its accepted output, derived from its generation, never stored separately.
- **Cached prefix**: an immutable resume state at the end of a path, with the generation method's
  checkpoint at that position.
- **Resume state**: what making a request resident at a position needs and nothing more: the
  numerical state of every sequence lane, and encoded features of the media spans that straddle
  the position. It holds no request's input.
- **Hit**: the result of matching a path against the cache. It exists only when the entry's path
  equals the queried path up to its position, at an exact boundary of the queried path.

One cache belongs to one loaded model, so artifact, tokenizer and state codec are fixed for every
path it holds and are not part of an entry's identity.

## Ownership

| Fact | Sole owner |
|---|---|
| Which tokens and media a cached state represents | The cached prefix's path |
| Whether a request may resume from cached state | Cache lookup |
| A request's tokens, layout and accepted output | Its generation |
| A request's prepared input and media | The executor, for the request's whole lifetime |
| A resident request's numerical state and encoded features | The executor, for its residency |
| When to cache (branch point, prompt boundary, terminal) and when to wait for a peer | Scheduler policy |

## Lifecycle

- **Input** is installed once at admission and released only when the request closes.
- **Residency** comes and goes within that lifetime. Becoming resident is one transition: look up
  the deepest cached prefix of the request's path below its resume bound, fork its state (or
  create fresh state), adopt cached features for straddling spans, encode once each image placed
  by a span that ends after the position and still lacks features (placements of one image share
  its features), and set the generation's position.
- A **newly admitted request**, a **request waiting for a peer's prefix**, and an **evicted
  request** all become resident through that one transition, when rounds are formed. They differ
  only in when it runs.
- **Eviction** releases numerical state and encoded features; the input stays, so replay is
  conditioned exactly as the first pass.
- **Resume bound**: a resumed request still computes the row it next samples from, so only
  prefixes strictly below it qualify: the prompt length before the first token, the accepted
  length after.

Caching points: a planned branch point where a request diverges from a cached path or a live
peer's prompt; one row before the prompt end; and a finished request's whole path, prompt and
reply, so the next turn of a conversation resumes at its end. A request whose path matches a live
peer's prompt beyond anything cached waits, holding no numerical state, until the peer caches the
shared prefix or stops computing it; the prefix is computed once.

## Invariants

1. A hit exists only for an entry whose path is a prefix of the queried path at an exact
   boundary.
2. Making a request resident from a hit cannot fail for input reasons. A shortfall of capacity
   gives the request a capacity wait at the current availability epoch, ended when the epoch
   advances. When, between flights, memory is Normal and no request awaits host credit or memory
   or has finished with state still to release, nothing can advance the epoch: every capacity
   wait then fails with its typed capacity error ([scheduler](scheduler.md)). In Reclaim, or
   without a memory reading, residency waits until memory is Normal again. Any other failure is
   an engine invariant failure, never a request error.
3. A request with installed input keeps it until it closes; every row the executor lowers has its
   input. There is no fallback for missing input.
4. A resume state references features only for spans that straddle its position.
5. Classified holdings count each feature allocation once, across request inputs and cached
   prefixes.
6. For any sequence of requests (interleaved, waiting, evicted, cancelled), a request whose live
   path shares a cached prefix of at least the minimum hit length, at an exact boundary below its
   resume bound, resumes from the deepest such prefix.

## Acceptance criteria

- A conversation's next turn whose prompt extends the previous turn's prompt and reply resumes at
  the end of that reply, with output identical to a cold run.
- A next turn whose template rewrites the previous reply resumes at the previous prompt boundary.
- An evicted request with media replays to output identical to an uninterrupted run.
- A waiting request resumes from its peer's prefix without recomputing or re-encoding it.
- Every charged byte stays attributed while prefixes are cached, used and evicted.
