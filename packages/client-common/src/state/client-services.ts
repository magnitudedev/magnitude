import { Context, Layer } from "effect"
import { Files, FilesLive } from "../files/service"
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
import { ProjectFiles, ProjectFilesLive } from "../project-files/service"
import { ChangesLive } from "./changes"
import { ClientEffectQuery } from "./client-effect-query"

export type ClientServices =
  | ClientEffectQuery
  | Files
  | LocalModels
  | ModelSlots
  | OnboardingPersistence
  | OnboardingModelSetup
  | ProjectFiles

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
    FilesLive,
    LocalModelsLive,
    ModelSlotsLive,
    OnboardingPersistenceLive,
    ProjectFilesLive,
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
