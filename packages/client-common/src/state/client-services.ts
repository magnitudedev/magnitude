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
import { ChangesLive } from "./changes"
import { ClientEffectQuery } from "./client-effect-query"

export type ClientServices =
  | ClientEffectQuery
  | LocalModels
  | ModelSlots
  | OnboardingPersistence
  | OnboardingModelSetup

export interface ClientServicesOptions {
  readonly onboardingSetupInitiallyOpen?: boolean
}

export const clientServicesLayer = (
  effectQuery: Context.Tag.Service<typeof ClientEffectQuery>,
  options: ClientServicesOptions = {},
) => {
  const infrastructure = Layer.succeed(ClientEffectQuery, effectQuery)
  const domains = Layer.mergeAll(
    ChangesLive,
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
