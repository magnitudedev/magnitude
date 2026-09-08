# Component and dimension index

Definitions live with their component owners. This index contains identities and
links only; [performance](../performance.md) defines evaluation and evidence rules.
Reusable mathematics lives in [resources](derivations/resources.md),
[neural operations](derivations/neural.md), [state](derivations/state.md) and
[service](derivations/service.md). Executable dimension definitions and dispatch live in
[theory/catalog.py](../../performance/theory/catalog.py). The component records explain
contracts and assumptions; current calculations come from those functions.

## Shared model contracts

| Component type | Dimensions |
|---|---|
| [MODEL:EMBEDDING](../models/composability.md#modelembedding) | [MODEL:EMBEDDING/EXEC](../models/composability.md#modelembedding) |
| [MODEL:EXPERTS](../models/composability.md#modelexperts) | [MODEL:EXPERTS/EXEC](../models/composability.md#modelexperts) |
| [MODEL:ATTENTION](../models/composability.md#modelattention) | [MODEL:ATTENTION/EXEC](../models/composability.md#modelattention) |
| [MODEL:GATED_DELTA](../models/composability.md#modelgated_delta) | [MODEL:GATED_DELTA/EXEC](../models/composability.md#modelgated_delta) |
| [MODEL:INPUT_FEATURES](../models/inputs.md#modelinput_features) | [MODEL:INPUT_FEATURES/EXEC](../models/inputs.md#modelinput_features) |

## Generic upstream contracts

| Component type | Dimensions |
|---|---|
| [MODEL:EXECUTOR](../models/architectures/generic-mlx-vlm.md#modelexecutor) | [MODEL:EXECUTOR/EXEC](../models/architectures/generic-mlx-vlm.md#modelexecutor) |
| [MODEL:LOADING](../models/architectures/generic-mlx-vlm.md#modelloading) | [MODEL:LOADING/LAT](../models/architectures/generic-mlx-vlm.md#modelloading), [MODEL:LOADING/MEM](../models/architectures/generic-mlx-vlm.md#modelloading) |
| [MODEL:FORWARD](../models/architectures/generic-mlx-vlm.md#modelforward) | [MODEL:FORWARD/EXEC](../models/architectures/generic-mlx-vlm.md#modelforward) |
| [STATE:CHECKPOINTS](../models/architectures/generic-mlx-vlm.md#statecheckpoints) | [STATE:CHECKPOINTS/MEM](../models/architectures/generic-mlx-vlm.md#statecheckpoints), [STATE:CHECKPOINTS/RESTORE](../models/architectures/generic-mlx-vlm.md#statecheckpoints) |

## Qwen contracts

| Component type | Dimensions |
|---|---|
| [MODEL:QWEN35](../models/architectures/qwen35.md#modelqwen35) | [MODEL:QWEN35/EXEC](../models/architectures/qwen35.md#modelqwen35) |
| [MODEL:QWEN35.VISION](../models/architectures/qwen35.md#modelqwen35vision) | [MODEL:QWEN35.VISION/EXEC](../models/architectures/qwen35.md#modelqwen35vision) |
| [MODEL:QWEN35.ATTENTION](../models/architectures/qwen35.md#modelqwen35attention) | [MODEL:QWEN35.ATTENTION/EXEC](../models/architectures/qwen35.md#modelqwen35attention) |
| [MODEL:QWEN35.RECURRENCE](../models/architectures/qwen35.md#modelqwen35recurrence) | [MODEL:QWEN35.RECURRENCE/EXEC](../models/architectures/qwen35.md#modelqwen35recurrence) |
| [MODEL:QWEN35.FEEDFORWARD](../models/architectures/qwen35.md#modelqwen35feedforward) | [MODEL:QWEN35.FEEDFORWARD/EXEC](../models/architectures/qwen35.md#modelqwen35feedforward) |
| [MODEL:QWEN35.READOUT](../models/architectures/qwen35.md#modelqwen35readout) | [MODEL:QWEN35.READOUT/EXEC](../models/architectures/qwen35.md#modelqwen35readout) |
| [STATE:QWEN35](../models/architectures/qwen35.md#stateqwen35) | [STATE:QWEN35/MEM](../models/architectures/qwen35.md#stateqwen35), [STATE:QWEN35/RESTORE](../models/architectures/qwen35.md#stateqwen35) |
| [MODEL:QWEN35.MTP](../models/architectures/qwen35.md#modelqwen35mtp) | [MODEL:QWEN35.MTP/EXEC](../models/architectures/qwen35.md#modelqwen35mtp) |

## Gemma contracts

| Component type | Dimensions |
|---|---|
| [MODEL:GEMMA4](../models/architectures/gemma4.md#modelgemma4) | [MODEL:GEMMA4/EXEC](../models/architectures/gemma4.md#modelgemma4) |
| [MODEL:GEMMA4.VISION](../models/architectures/gemma4.md#modelgemma4vision) | [MODEL:GEMMA4.VISION/EXEC](../models/architectures/gemma4.md#modelgemma4vision) |
| [MODEL:GEMMA4.ATTENTION](../models/architectures/gemma4.md#modelgemma4attention) | [MODEL:GEMMA4.ATTENTION/EXEC](../models/architectures/gemma4.md#modelgemma4attention) |
| [MODEL:GEMMA4.KV](../models/architectures/gemma4.md#modelgemma4kv) | [MODEL:GEMMA4.KV/EXEC](../models/architectures/gemma4.md#modelgemma4kv) |
| [MODEL:GEMMA4.FEEDFORWARD](../models/architectures/gemma4.md#modelgemma4feedforward) | [MODEL:GEMMA4.FEEDFORWARD/EXEC](../models/architectures/gemma4.md#modelgemma4feedforward) |
| [MODEL:GEMMA4.MLP](../models/architectures/gemma4.md#modelgemma4mlp) | [MODEL:GEMMA4.MLP/EXEC](../models/architectures/gemma4.md#modelgemma4mlp) |
| [MODEL:GEMMA4.EXPERT_BRANCH](../models/architectures/gemma4.md#modelgemma4expert_branch) | [MODEL:GEMMA4.EXPERT_BRANCH/EXEC](../models/architectures/gemma4.md#modelgemma4expert_branch) |
| [MODEL:GEMMA4.INPUTS](../models/architectures/gemma4.md#modelgemma4inputs) | [MODEL:GEMMA4.INPUTS/EXEC](../models/architectures/gemma4.md#modelgemma4inputs) |
| [MODEL:GEMMA4.READOUT](../models/architectures/gemma4.md#modelgemma4readout) | [MODEL:GEMMA4.READOUT/EXEC](../models/architectures/gemma4.md#modelgemma4readout) |

## Engine contracts

| Component type | Dimensions |
|---|---|
| [ENGINE:INFERENCE](../engine/components.md#engineinference) | [ENGINE:INFERENCE/RATE](../engine/components.md#engineinference), [ENGINE:INFERENCE/TTFT](../engine/components.md#engineinference), [ENGINE:INFERENCE/GAP](../engine/components.md#engineinference) |
| [SCHEDULING:ADMISSION](../engine/components.md#schedulingadmission) | [SCHEDULING:ADMISSION/LAT](../engine/components.md#schedulingadmission) |
| [SCHEDULING:SERVICE](../engine/components.md#schedulingservice) | [SCHEDULING:SERVICE/RATE](../engine/components.md#schedulingservice), [SCHEDULING:SERVICE/TTFT](../engine/components.md#schedulingservice), [SCHEDULING:SERVICE/GAP](../engine/components.md#schedulingservice) |
| [SCHEDULING:PREFILL](../engine/components.md#schedulingprefill) | [SCHEDULING:PREFILL/EXEC](../engine/components.md#schedulingprefill) |
| [BATCHING:ASSEMBLY](../engine/components.md#batchingassembly) | [BATCHING:ASSEMBLY/EXEC](../engine/components.md#batchingassembly) |
| [EXECUTION:DEVICE](../engine/components.md#executiondevice) | [EXECUTION:DEVICE/EXEC](../engine/components.md#executiondevice) |
| [MEMORY:ACCOUNTING](../engine/components.md#memoryaccounting) | [MEMORY:ACCOUNTING/EXEC](../engine/components.md#memoryaccounting) |
| [CACHE:PREFIX](../engine/components.md#cacheprefix) | [CACHE:PREFIX/REUSE](../engine/components.md#cacheprefix) |
| [KV:STORE](../engine/components.md#kvstore) | [KV:STORE/MEM](../engine/components.md#kvstore) |
| [KV:APPEND](../engine/components.md#kvappend) | [KV:APPEND/EXEC](../engine/components.md#kvappend) |
| [KV:BRANCH](../engine/components.md#kvbranch) | [KV:BRANCH/EXEC](../engine/components.md#kvbranch) |
| [STATE:RECURRENT](../engine/components.md#staterecurrent) | [STATE:RECURRENT/MEM](../engine/components.md#staterecurrent), [STATE:RECURRENT/RESTORE](../engine/components.md#staterecurrent) |
| [GENERATION:PLAIN](../engine/components.md#generationplain) | [GENERATION:PLAIN/EXEC](../engine/components.md#generationplain) |
| [GENERATION:SPECULATION](../engine/components.md#generationspeculation) | [GENERATION:SPECULATION/EXEC](../engine/components.md#generationspeculation) |
| [GENERATION:SAMPLING](../engine/components.md#generationsampling) | [GENERATION:SAMPLING/EXEC](../engine/components.md#generationsampling) |
| [GENERATION:ACCEPTANCE](../engine/components.md#generationacceptance) | [GENERATION:ACCEPTANCE/EXEC](../engine/components.md#generationacceptance) |
