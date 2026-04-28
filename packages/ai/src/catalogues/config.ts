import { Context, Effect } from "effect"

export interface DiscoveredModel {
  readonly id: string
  readonly name?: string
  readonly maxContextTokens: number | null
  readonly discoveredAt?: string
  readonly source?: string
}

export type DiscoveryStatus = "success_non_empty" | "success_empty" | "failure"

export interface ProviderOptions {
  readonly baseUrl?: string
  readonly rememberedModelIds?: readonly string[]
  readonly discoveredModels?: readonly DiscoveredModel[]
  readonly inventoryUpdatedAt?: string
  readonly lastDiscoveryStatus?: DiscoveryStatus
  readonly lastDiscoverySource?: string
  readonly lastDiscoveryDiagnostics?: readonly string[]
  readonly lastDiscoveryError?: string
}

export class CatalogueConfig extends Context.Tag("@magnitudedev/ai/CatalogueConfig")<
  CatalogueConfig,
  {
    readonly getProviderOptions: (providerId: string) => Effect.Effect<ProviderOptions | undefined>
    readonly setProviderOptions: (
      providerId: string,
      optionsOrUpdater:
        | ProviderOptions
        | undefined
        | ((current: ProviderOptions | undefined) => ProviderOptions | undefined),
    ) => Effect.Effect<void>
  }
>() {}
