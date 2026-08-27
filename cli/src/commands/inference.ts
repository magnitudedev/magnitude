import type { Command as Commander } from "@commander-js/extra-typings"
import { Atom, Registry } from "@effect-atom/atom"
import { Client, Mutation } from "@magnitudedev/effect-query"
import {
  MagnitudeBoundary,
  type MagnitudeImplementationError,
  ProviderModelIdSchema,
  SlotIdSchema,
  installedAcquisition,
  magnitudeImplementationsLayer,
} from "@magnitudedev/sdk"
import { Effect, Option, Schema } from "effect"
import { makeTerminalPlatform } from "../platform/terminal"

const decodeModelId = Schema.decodeUnknown(ProviderModelIdSchema)
const decodeSlotId = Schema.decodeUnknown(SlotIdSchema)
type CliModelsClient = Pick<
  Client.Materialized<typeof MagnitudeBoundary, unknown, MagnitudeImplementationError>,
  "Models"
>

const printJson = (value: unknown) => Effect.sync(() => {
  process.stdout.write(`${JSON.stringify(value, (_key, member) => {
    if (typeof member !== "object" || member === null || member._id !== "Option") return member
    return member._tag === "Some" ? member.value : undefined
  }, 2)}\n`)
})

const explain = (error: unknown): string => typeof error === "object"
  && error !== null
  && "message" in error
  && typeof error.message === "string"
  ? error.message
  : String(error)

const runModels = <Value>(
  use: (client: CliModelsClient, registry: Registry.Registry) => Effect.Effect<Value, unknown>,
) => Effect.runPromise(Effect.scoped(
  Effect.gen(function* () {
    const terminal = yield* makeTerminalPlatform({
      launchCommand: Option.none(),
      debug: false,
      effectLoggingLayer: Option.none(),
    })
    const registry = Registry.make()
    yield* Effect.addFinalizer(() => Effect.sync(() => registry.dispose()))
    const client = Client.make(
      MagnitudeBoundary,
      magnitudeImplementationsLayer(terminal.platform.protocolLayer),
    )
    return yield* use(client, registry)
  }).pipe(
    Effect.tap(printJson),
    Effect.catchAll((error) => Effect.sync(() => {
      process.stderr.write(`${explain(error)}\n`)
      process.exitCode = 1
    })),
    Effect.asVoid,
  ),
))

export const registerInferenceCommands = (program: Commander): void => {
  program.command("hardware")
    .description("Show the local inference environment")
    .action(() => runModels((client, registry) =>
      Registry.getResult(registry, Atom.make((get) => get(client.Models.GetLocalEnvironment({})).result))))

  const models = program.command("models").description("Inspect and manage models")
  models.command("list")
    .description("Show the unified model catalog")
    .action(() => runModels((client, registry) =>
      Registry.getResult(registry, Atom.make((get) => get(client.Models.GetCatalog({})).result))))
  models.command("install")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => runModels((client, registry) => decodeModelId(modelId).pipe(
      Effect.flatMap((decoded) => Mutation.execute(client.Models.SyncLocalModel, { modelId: decoded })),
      Effect.provideService(Registry.AtomRegistry, registry),
    )))
  models.command("remove")
    .argument("<model-id>", "Canonical model ID")
    .action((modelId) => runModels((client, registry) => decodeModelId(modelId).pipe(
      Effect.flatMap((decoded) => Mutation.execute(client.Models.RemoveLocalModel, { modelId: decoded })),
      Effect.provideService(Registry.AtomRegistry, registry),
    )))

  const downloads = program.command("downloads").description("Inspect and control model downloads")
  downloads.command("list")
    .description("Show download state in the unified model catalog")
    .action(() => runModels((client, registry) =>
      Registry.getResult(registry, Atom.make((get) => get(client.Models.GetCatalog({})).result))))
  downloads.command("cancel")
    .argument("<model-id>", "Canonical model ID whose download to cancel")
    .action((modelId) => runModels((client, registry) => decodeModelId(modelId).pipe(
      Effect.flatMap((decoded) => Mutation.execute(client.Models.CancelLocalModelSync, { modelId: decoded })),
      Effect.provideService(Registry.AtomRegistry, registry),
    )))
  downloads.command("acknowledge-failure")
    .argument("<model-id>", "Canonical model ID whose failed download to acknowledge")
    .action((modelId) => runModels((client, registry) => decodeModelId(modelId).pipe(
      Effect.flatMap((decoded) => Mutation.execute(client.Models.AcknowledgeLocalModelSyncFailure, { modelId: decoded })),
      Effect.provideService(Registry.AtomRegistry, registry),
    )))

  const instances = program.command("instances").description("Inspect and control model residency")
  instances.command("list")
    .action(() => runModels((client, registry) =>
      Registry.getResult(registry, Atom.make((get) => get(client.Models.GetCatalog({})).result)).pipe(
        Effect.map((state) => state._tag === "Initializing"
          ? { models: [] }
          : {
              models: state.models.flatMap((entry) => {
                if (entry._tag !== "Local") return []
                const installed = installedAcquisition(entry.product.acquisitionState)
                return installed === undefined
                  ? []
                  : [{ modelId: entry.product.modelId, residencyState: installed.residencyState }]
              }),
            }),
      )))
  instances.command("load")
    .argument("<slot-id>", "Configured slot ID")
    .action((slotId) => runModels((client, registry) => decodeSlotId(slotId).pipe(
      Effect.flatMap((decoded) => Mutation.execute(client.Models.LoadSlot, { slotId: decoded })),
      Effect.provideService(Registry.AtomRegistry, registry),
    )))
  instances.command("stop")
    .argument("<slot-id>", "Configured slot ID")
    .action((slotId) => runModels((client, registry) => decodeSlotId(slotId).pipe(
      Effect.flatMap((decoded) => Mutation.execute(client.Models.StopSlot, { slotId: decoded })),
      Effect.provideService(Registry.AtomRegistry, registry),
    )))

  program.command("slots")
    .description("Show resolved model slots")
    .action(() => runModels((client, registry) =>
      Registry.getResult(registry, Atom.make((get) => get(client.Models.GetSlots({})).result))))
}
