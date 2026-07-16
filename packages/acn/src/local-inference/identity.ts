import { createHash } from "node:crypto"
import type { LlamaCpp } from "@magnitudedev/local-inference"
import { ProviderModelIdSchema, type ProviderModelId } from "@magnitudedev/sdk"

export const providerModelIdForModelPath = (path: LlamaCpp.NormalizedLlamaModelPath): ProviderModelId =>
  ProviderModelIdSchema.make(`lmp_${createHash("sha256").update("llamacpp-model-path-v1\0").update(String(path)).digest("hex")}`)

export const isOpaqueLlamaRoutingName = (value: string): boolean =>
  /^lmp_[a-f0-9]{64}$/i.test(value)
