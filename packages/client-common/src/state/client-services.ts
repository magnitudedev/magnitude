import { ApplicationSession, ApplicationSessionLive } from "../application/session"
import { ApplicationHost, ApplicationHostFailed } from "../application/host"
import { ApplicationRouter, ApplicationRouterLocation, ApplicationRouterMemory } from "../application/router"
import { Context, Effect, Layer, Option } from "effect"
import { Files, FilesLive } from "../files/service"
import { LocalModels, LocalModelsLive } from "../local-models/service"
import { ModelSlots, ModelSlotsLive } from "../model-slots/service"
import { ProjectFiles, ProjectFilesLive } from "../project-files/service"
import { ChangesLive } from "./changes"
import { ClientEffectQuery } from "./client-effect-query"

export type ClientServices =
  | ApplicationSession
  | ApplicationRouter
  | ClientEffectQuery
  | Files
  | LocalModels
  | ModelSlots
  | ProjectFiles

export interface ClientServicesOptions {
  readonly host?: ApplicationHost
  readonly navigation?: "memory" | "location"
}

export const clientServicesLayer = (
  effectQuery: Context.Tag.Service<typeof ClientEffectQuery>,
  options: ClientServicesOptions = {},
) => {
  const infrastructure = Layer.succeed(ClientEffectQuery, effectQuery)
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
  ).pipe(
    Layer.provideMerge(observedInfrastructure),
  )

  const router = options.navigation === "location" ? ApplicationRouterLocation : ApplicationRouterMemory
  return ApplicationSessionLive.pipe(
    Layer.provideMerge(router),
    Layer.provideMerge(domains),
    Layer.provide(Layer.succeed(ApplicationHost, options.host ?? hostlessApplication)),
  )
}

/** A client with no application host, such as the CLI: no window, shell, or desktop controls. */
export const hostlessApplication: ApplicationHost = {
  window: Option.none(),
  appearance: {
    read: Effect.succeed("system"),
    save: () => Effect.fail(new ApplicationHostFailed({ message: "Appearance can't be saved here." })),
  },
  shell: Option.none(),
  desktop: Option.none(),
}
