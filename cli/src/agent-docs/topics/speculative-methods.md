# Speculative decoding methods

Speculative decoding speeds up generation by using a cheaper mechanism to propose several future
tokens, then asking the full target model to verify those tokens together. The target still decides
which tokens are accepted. These methods affect decode speed, not the model's intelligence or
quality rating.

## Practical hierarchy

For the methods Magnitude reports, use this rule of thumb for decode speed:

```text
None (usually slowest) -> MTP -> DFlash -> DSpark (usually fastest)
```

This is a typical ordering, not a guarantee. Actual speed depends on the target and draft models,
draft acceptance, prompt and output content, context length, quantization, hardware, memory
placement, and request concurrency. A speculative method can provide little benefit or even add
overhead in an unfavorable configuration. Prefer Magnitude's model- and machine-specific speed
evidence over the method name alone.

## Methods

- **None**: The target model performs ordinary autoregressive decoding without a speculative draft.
  It normally completes one target-approved token per decode step and is the baseline against which
  speculative acceleration is measured.
- **MTP**: Multi-Token Prediction, also called NextN, uses auxiliary modules trained with the target
  model to propose future tokens more cheaply than the full target. These modules are commonly
  embedded in or closely coupled to the target. MTP reduces full-model decode work, but its useful
  draft is generally shorter or more sequential than the block-parallel methods below.
- **DFlash**: A lightweight, target-specific block-diffusion drafter uses hidden features from the
  target model and proposes an entire token block in one forward pass. Parallel block drafting
  amortizes draft latency and maps well to accelerators, so DFlash is typically faster than MTP.
  Because its block positions have weaker left-to-right dependencies, acceptance can fall toward
  the end of a proposed block.
- **DSpark**: Extends DFlash's parallel backbone with a lightweight semi-autoregressive, typically
  Markov, head that restores dependencies between tokens inside the proposed block. A confidence
  head can also trim low-confidence suffix tokens before verification. This retains most of the
  parallel drafting speed while producing longer useful prefixes and avoiding wasted target
  verification, so DSpark is typically faster than DFlash.

## What Magnitude handles

For a catalog model, Magnitude's reviewed configuration declares the exact speculative method and
any required draft material. `magnitude catalog pull` acquires a separate draft artifact when the
configuration needs one. Assessment and loading validate the target, draft, method, hardware fit,
and serving configuration; Magnitude then activates the method automatically during inference.

The user does not need to select a method or pair a draft model manually. Magnitude does not attach
an arbitrary draft model to a target. If a catalog configuration has no reviewed compatible
speculative method, it runs without speculative decoding.
