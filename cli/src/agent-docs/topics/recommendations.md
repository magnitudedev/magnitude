# Model recommendations

`magnitude models recommend` lists the models Magnitude has tested, in the order it
recommends them for this computer. Each tested model has a ladder of quants, smallest to largest,
and each recommendation names the quant chosen for this computer: the largest one that fits in
memory, has room on disk and is estimated to decode at 16 tokens per second or more in a long agent
conversation; else the largest within 90% of the fastest that fits; and where no speed can be
measured, the smallest that fits.

Models whose chosen quant reaches 16 tokens per second come first, by intelligence; then those that
reach 10 tokens per second; then slower ones. Each group is ordered by intelligence, and models below
16 tok/s are marked "below 16 tok/s on this computer". `--country US` recommends only
models whose developer is from that country, choosing among them the same way. A recommendation is
a starting point for choosing with the user, not a claim that one model is best for every workload.

## What the displayed properties tell you

- **Quant** names the chosen quant by bit width, publisher and quant name, such as
  "4-bit unsloth UD-Q4_K_XL". Fewer bits mean a smaller, faster model that preserves the source
  model less faithfully.
- **Size** is the quant's download size.
- **Speed** is Magnitude's estimate of decode throughput on this computer in a long agent
  conversation, or at the served context when that is shorter. It is a hardware-aware prediction
  rather than a benchmark run of the downloaded model, and real speed can vary with workload and
  system activity. `magnitude models show <model-id>` gives the estimate at each assessed context
  depth.
- **Placement** says how the quant runs here: whole in GPU or unified memory, or with its experts
  streamed from host RAM or from disk. `magnitude models show <model-id>` gives the bytes on the
  device, in host RAM and read from disk. A model that normally fits can still need other
  memory-intensive applications to be closed before it loads.
- **Context** is the longest conversation this computer serves the model at. It is the model's own
  context when that fits in memory, and otherwise the largest power of two below it, down to 32K, that
  does, shown as "64K of 256K". Below 32K a model is reported as not fitting, since a coding agent's
  prompt and tools alone fill much of a smaller context. A request longer than the served context
  fails with `context_length_exceeded` rather than running the computer out of memory; connected
  harnesses are updated with the served context.
- **Intelligence** is how well the model does on today's hardest benchmarks, as a percentage of the
  best model in the world. 100% is the frontier model; 50% means the model is expected to score half
  of what the frontier model scores on those benchmarks. Quantized variants of the same source model
  share this model-level score.
- **Quants** is the model's tested ladder, smallest to largest, each rung with its size, speed,
  context, placement and model ID; `*` marks the chosen one.
- **Acceleration** identifies the speculative-decoding method Magnitude has prepared for that
  configuration. See `magnitude docs speculative-methods` for what `None`, `MTP`, `DFlash`,
  `DSpark`, `DFlash2` and `LiLiCorr` mean and how Magnitude sets them up.
- **Capabilities** identify supported features such as vision, tool use, structured output, and
  reasoning. These can matter more than a small speed or intelligence difference when the user has
  a specific task in mind.
- **Model ID** is the exact configuration identifier required by later commands. Preserve it exactly;
  the friendly model name is not a command argument.

Use these signals together. A faster model may feel better for quick iteration, while a more
intelligent model may be worth waiting for on difficult coding or reasoning tasks. Placement and
context determine whether the model is practical on this computer, the quant's bit width indicates
how much quality the local format may give up, and capabilities determine whether it can do the job
at all.

## How intelligence is measured

Magnitude's intelligence comes from [Epoch AI's Capabilities Index](https://epoch.ai/benchmarks/eci)
(ECI). Epoch rates models on dozens of benchmarks and fits how hard each benchmark is, which places
every model on one capability scale. From a model's place on that scale, Epoch's fit predicts its
score on each benchmark. Intelligence is the model's average predicted score on the benchmarks Epoch
rates that were released in the last two years, divided by the frontier model's. Older benchmarks
drop out as they age, so the measure keeps tracking what is hard today.

Each model's place on the scale comes from one of three sources:

- **Epoch published:** Epoch rates the model itself.
- **From Epoch's data:** Epoch has run the model on enough of its benchmarks, and Magnitude fits its
  place with Epoch's own benchmark difficulties.
- **Bridged:** Epoch has not run the model, so Magnitude predicts its place from other published
  evaluations, calibrated on models Epoch does rate. These carry the most uncertainty.

The frontier model and the benchmark window are reviewed with each release. Magnitude ships the
scores with the models it recommends rather than refreshing them from the network during
recommendation.
