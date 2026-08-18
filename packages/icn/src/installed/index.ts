import { Context, Duration, Effect, Layer, Stream } from "effect"
import type { InstalledModelPackagesResponse } from "@magnitudedev/icn-protocol/schemas"
import { IcnModels, type IcnModelsService } from "../models/index.js"
import type { IcnObservedSnapshot, IcnObservedState } from "../observed-state.js"

export interface IcnInstalledModelsService
  extends IcnObservedState<InstalledModelPackagesResponse, unknown> {}

export class IcnInstalledModels extends Context.Tag("@magnitudedev/icn/IcnInstalledModels")<
  IcnInstalledModels,
  IcnInstalledModelsService
>() {}

export interface IcnInstalledModelsOptions {
  readonly refreshInterval?: Duration.DurationInput
}

const installedSnapshot = (
  snapshot: Effect.Effect.Success<IcnModelsService["get"]>,
): IcnObservedSnapshot<InstalledModelPackagesResponse> => {
  const byId = new Map(snapshot.state.uncataloguedPackages.map((entry) => [entry.package.id, entry]))
  for (const model of snapshot.state.catalogModels) {
    if (model.localState._tag !== "Installed") continue
    for (const present of model.localState.installation.packages) {
      byId.set(present.package.id, present)
    }
  }
  return {
    revision: snapshot.revision,
    state: {
      revision: snapshot.state.revision,
      reconciliationComplete: snapshot.state.reconciliationComplete,
      packages: [...byId.values()],
    },
  }
}

export const makeIcnInstalledModels = (
  _options: IcnInstalledModelsOptions = {},
): Layer.Layer<IcnInstalledModels, never, IcnModels> =>
  Layer.effect(
    IcnInstalledModels,
    Effect.gen(function* () {
      const models = yield* IcnModels
      return IcnInstalledModels.of({
        get: models.get.pipe(Effect.map(installedSnapshot)),
        changes: models.changes.pipe(Stream.map(installedSnapshot)),
        initialized: models.initialized,
        refresh: models.refresh,
      })
    }),
  )
