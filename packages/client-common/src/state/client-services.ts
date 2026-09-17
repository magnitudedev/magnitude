import { DesktopBridge, DesktopSession, DesktopSessionLive } from "../desktop/service"
import { Context, Layer, Option } from "effect"
import { Files, FilesLive } from "../files/service"
import { LocalModels, LocalModelsLive } from "../local-models/service"
import { ModelSlots, ModelSlotsLive } from "../model-slots/service"
import { ProjectFiles, ProjectFilesLive } from "../project-files/service"
import { ChangesLive } from "./changes"
import { ClientEffectQuery } from "./client-effect-query"
import {
  HarnessConnection,
  UnavailableHarnessConnection,
} from "../harness-connections/service"

export type ClientServices =
  | DesktopSession
  | ClientEffectQuery
  | Files
  | LocalModels
  | ModelSlots
  | ProjectFiles
  | HarnessConnection

export interface ClientServicesOptions {
  readonly desktopBridge?: DesktopBridge
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
    ProjectFilesLive,
    harnessConnection,
  ).pipe(
    Layer.provideMerge(observedInfrastructure),
  )

  return DesktopSessionLive.pipe(
    Layer.provideMerge(domains),
    Layer.provide(Layer.succeed(DesktopBridge, Option.fromNullable(options.desktopBridge))),
  )
}
