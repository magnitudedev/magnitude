import type { LocalModel, MagnitudeClient, ModelCatalogState } from "@magnitudedev/sdk"
import { Context, Effect, type Scope } from "effect"
import { existingAcnConnection } from "../server/acn-connection"

export type CliModelsClient = Pick<MagnitudeClient, "models">

/** How inference commands reach the service; by default the running Magnitude service, acquired only when a command needs it. */
export class CliModelsConnection extends Context.Reference<CliModelsConnection>()("CliModelsConnection", {
  defaultValue: (): { readonly connect: Effect.Effect<CliModelsClient, unknown, Scope.Scope> } => ({
    connect: Effect.map(existingAcnConnection, ({ client }) => client),
  }),
}) {}

export const withClient = <A>(use: (client: CliModelsClient) => Effect.Effect<A, unknown>) =>
  Effect.scoped(Effect.gen(function* () {
    const { connect } = yield* CliModelsConnection
    return yield* use(yield* connect)
  }))

export const readCatalog = (client: CliModelsClient) => client.models.getCatalog({})
export const readHardware = (client: CliModelsClient) => client.models.getLocalEnvironment({})

/** Every local product the catalog lists: catalog models and visible custom models. */
export const catalogProducts = (catalog: ModelCatalogState): readonly LocalModel[] =>
  catalog._tag === "Initializing" ? [] : catalog.models.flatMap((entry) => entry._tag === "Local" ? [entry.product] : [])

/** Every model a command can address by ID: the visible ones and the ones the person hid. */
export const addressableModels = (catalog: ModelCatalogState): readonly LocalModel[] =>
  catalog._tag === "Initializing" ? [] : [...catalogProducts(catalog), ...catalog.hiddenLocalModels]
