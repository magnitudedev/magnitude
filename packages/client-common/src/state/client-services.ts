import { Context, Layer } from "effect"
import { LocalModels, LocalModelsLive } from "../local-models/service"
import {
  OnboardingModelSetup,
  OnboardingModelSetupConfig,
  OnboardingModelSetupLive,
} from "../local-models/setup"
import { ModelSlots, ModelSlotsLive } from "../model-slots/service"
import {
  OnboardingPersistence,
  OnboardingPersistenceLive,
} from "../onboarding/persistence"
import { ClientEffectQuery } from "./client-effect-query"
import {
  ClientInvalidations,
} from "./client-invalidations"

export type ClientServices =
  | ClientEffectQuery
  | ClientInvalidations
  | LocalModels
  | ModelSlots
  | OnboardingPersistence
  | OnboardingModelSetup

export interface ClientServicesOptions {
  readonly onboardingSetupInitiallyOpen?: boolean
}

export const clientServicesLayer = (
  effectQuery: Context.Tag.Service<typeof ClientEffectQuery>,
  invalidations: Context.Tag.Service<typeof ClientInvalidations>,
  options: ClientServicesOptions = {},
) => {
  const infrastructure = Layer.mergeAll(
    Layer.succeed(ClientEffectQuery, effectQuery),
    Layer.succeed(ClientInvalidations, invalidations),
  )
  const domains = Layer.mergeAll(
    LocalModelsLive,
    ModelSlotsLive,
    OnboardingPersistenceLive,
  ).pipe(
    Layer.provideMerge(infrastructure),
  )

  return OnboardingModelSetupLive.pipe(
    Layer.provideMerge(domains),
    Layer.provide(Layer.succeed(OnboardingModelSetupConfig, {
      initiallyOpen: options.onboardingSetupInitiallyOpen ?? false,
    })),
  )
}
