import { Context, Effect } from 'effect'
import type { ProviderDefinition, ModelDefinition, AuthInfo, OAuthAuth } from '../types'
import type { CallUsage, ModelSlot, SlotState, SlotUsage } from '../state/provider-state'
import type { DetectedAuthMethod, ProviderAuthMethodStatus } from '../detect'
import type { Model } from '../model/model'

export interface ProviderCatalogShape {
  readonly listProviders: () => Effect.Effect<readonly ProviderDefinition[]>
  readonly getProvider: (providerId: string) => Effect.Effect<ProviderDefinition | null>
  readonly getProviderName: (providerId: string) => Effect.Effect<string>
  readonly listModels: (providerId: string) => Effect.Effect<readonly ModelDefinition[]>
  readonly getModel: (providerId: string, modelId: string) => Effect.Effect<ModelDefinition | null>
  readonly refresh: () => Effect.Effect<void>
}

export class ProviderCatalog extends Context.Tag('ProviderCatalog')<
  ProviderCatalog,
  ProviderCatalogShape
>() {}

export interface ProviderStateShape {
  readonly peek: (slot?: ModelSlot) => Effect.Effect<{ model: Model; auth: AuthInfo | null } | null>
  readonly getSlot: (slot: ModelSlot) => Effect.Effect<SlotState>
  readonly setSelection: (
    slot: ModelSlot,
    providerId: string,
    modelId: string,
    auth: AuthInfo | null,
    options?: { readonly persist?: boolean },
  ) => Effect.Effect<boolean>
  readonly clear: (slot: ModelSlot) => Effect.Effect<void>
  readonly contextWindow: (slot?: ModelSlot) => Effect.Effect<number>
  readonly contextLimits: (slot?: ModelSlot) => Effect.Effect<{ hardCap: number; softCap: number }>
  readonly accumulateUsage: (slot: ModelSlot, usage: CallUsage) => Effect.Effect<void>
  readonly getUsage: (slot: ModelSlot) => Effect.Effect<SlotUsage>
  readonly resetUsage: (slot: ModelSlot) => Effect.Effect<void>
}

export class ProviderState extends Context.Tag('ProviderState')<
  ProviderState,
  ProviderStateShape
>() {}


export interface ProviderAuthShape {
  readonly loadAuth: () => Effect.Effect<Record<string, AuthInfo>>
  readonly getAuth: (providerId: string) => Effect.Effect<AuthInfo | undefined>
  readonly setAuth: (providerId: string, auth: AuthInfo) => Effect.Effect<void>
  readonly removeAuth: (providerId: string) => Effect.Effect<void>
  readonly refresh: (providerId: string, refreshToken: string) => Effect.Effect<OAuthAuth | null, Error>
  readonly detectProviders: () => Effect.Effect<readonly { provider: ProviderDefinition; authMethods: readonly DetectedAuthMethod[] }[]>
  readonly detectDefaultProvider: () => Effect.Effect<string | null>
  readonly detectProviderAuthMethods: (providerId: string) => Effect.Effect<ProviderAuthMethodStatus | null>
  readonly connectedProviderIds: () => Effect.Effect<Set<string>>
}

export class ProviderAuth extends Context.Tag('ProviderAuth')<
  ProviderAuth,
  ProviderAuthShape
>() {}