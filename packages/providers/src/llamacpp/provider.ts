import { Effect, Option } from "effect"
import type {
  BoundModel,
  ModelCatalog,
  Provider,
  BaseCallOptions,
  ProviderModelBindOptions,
} from "@magnitudedev/ai"
import { createLlamaCppCatalog } from "./catalog"
import { createLlamaCppCompatibleSpec, wrapAsBaseModel } from "./models"
import type { LlamaCppCallOptions, LlamaCppModelInfo } from "./contract"
import { classifyModelFamily as classifyModelFamilyRaw } from "../family-registry"

export const PROVIDER_ID = "llamacpp" as const

export interface LlamaCppClientConfig {
  readonly endpoint?: string
  readonly apiKey?: string
  readonly sessionId?: string
  readonly auth?: (headers: Headers) => void
}

const DEFAULT_ENDPOINT = "http://localhost:8080"

export type LlamaCppProvider = Provider<LlamaCppModelInfo>

export interface LlamaCppProviderInstance {
  readonly provider: LlamaCppProvider
  readonly catalog: ModelCatalog<LlamaCppModelInfo>
}

export function createLlamaCppProvider(config?: LlamaCppClientConfig): LlamaCppProviderInstance {
  const endpoint = config?.endpoint ?? process.env.LLAMACPP_ENDPOINT ?? DEFAULT_ENDPOINT

  const auth = config?.auth ?? (() => {
    const apiKey = config?.apiKey ?? process.env.LLAMACPP_API_KEY
    if (!apiKey) return undefined
    return (headers: Headers) => {
      headers.set("Authorization", `Bearer ${apiKey}`)
    }
  })()

  const classifyModelFamily = (model: Omit<LlamaCppModelInfo, "modelFamilyId">): Option.Option<string> =>
    classifyModelFamilyRaw(model.providerModelId)

  const catalog = createLlamaCppCatalog({
    endpoint,
    auth: auth ?? ((_: Headers) => {}),
    classify: classifyModelFamily,
  })

  const bindModel = (
    providerModelId: string,
    options?: ProviderModelBindOptions,
  ): Effect.Effect<BoundModel<BaseCallOptions>, never, never> =>
    Effect.succeed(
      wrapAsBaseModel(
        createLlamaCppCompatibleSpec({ modelId: providerModelId, endpoint }).bind({
          auth: auth ?? (() => {}),
          defaults: options?.defaults as Partial<LlamaCppCallOptions> | undefined,
          ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
        }),
      ),
    )

  const provider: LlamaCppProvider = {
    id: PROVIDER_ID,
    displayName: "Llama.cpp",
    catalog,
    bindModel,
    classifyModelFamily,
  }

  return { provider, catalog }
}
