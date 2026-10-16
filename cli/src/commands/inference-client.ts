import type { LocalModel, MagnitudeClient, ModelCatalogState } from "@magnitudedev/sdk"
import { Effect } from "effect"
import { existingAcnConnection } from "../server/acn-connection"

export type CliModelsClient = Pick<MagnitudeClient, "models">

export const withClient = <A>(use: (client: CliModelsClient) => Effect.Effect<A, unknown>) =>
  Effect.scoped(Effect.gen(function* () {
    const connection = yield* existingAcnConnection
    return yield* use(connection.client)
  }))

export const readCatalog = (client: CliModelsClient) => client.models.getCatalog({})
export const readHardware = (client: CliModelsClient) => client.models.getLocalEnvironment({})

/** Every local product the catalog lists: catalog models and visible custom models. */
export const catalogProducts = (catalog: ModelCatalogState): readonly LocalModel[] =>
  catalog._tag === "Initializing" ? [] : catalog.models.flatMap((entry) => entry._tag === "Local" ? [entry.product] : [])
