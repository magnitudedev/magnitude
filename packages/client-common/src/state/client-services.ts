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
import {
  HarnessConnection,
  UnavailableHarnessConnection,
} from "../harness-connections/service"

export type ClientServices =
  | ClientEffectQuery
  | Files
  | LocalModels
  | ModelSlots
  | OnboardingPersistence
  | OnboardingModelSetup
  | ProjectFiles
  | HarnessConnection

export interface ClientServicesOptions {
  readonly onboardingSetupInitiallyOpen?: boolean
  readonly harnessConnection?: HarnessConnection
}

export const clientServicesLayer = (
  effectQuery: Context.Tag.Service<typeof ClientEffectQuery>,
  options: ClientServicesOptions = {},
) => {
  const infrastructure = Layer.succeed(ClientEffectQuery, effectQuery)
  const harnessConnection = Layer.succeed(
    HarnessConnection,
    options.harnessConnection ?? UnavailableHarnessConnection,
  )
  // Establish the ACN change drain before any domain service performs its first
  // Query. This closes the read-before-watch startup race while retaining one
  // connection-scoped Effect Query runtime.
  const observedInfrastructure = ChangesLive.pipe(
    Layer.provideMerge(infrastructure),
  )
  const domains = Layer.mergeAll(
    FilesLive,
    LocalModelsLive,
    ModelSlotsLive,
    OnboardingPersistenceLive,
    ProjectFilesLive,
    harnessConnection,
  ).pipe(
    Layer.provideMerge(observedInfrastructure),
  )

  return OnboardingModelSetupLive.pipe(
    Layer.provideMerge(domains),
    Layer.provide(Layer.succeed(OnboardingModelSetupConfig, {
      initiallyOpen: options.onboardingSetupInitiallyOpen ?? false,
    })),
  )
}
