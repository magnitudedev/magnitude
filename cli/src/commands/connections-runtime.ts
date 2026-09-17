import { FetchHttpClient } from "@effect/platform"
import type * as CommandExecutor from "@effect/platform/CommandExecutor"
import type * as FileSystem from "@effect/platform/FileSystem"
import type * as HttpClient from "@effect/platform/HttpClient"
import type * as Path from "@effect/platform/Path"
import { BunContext } from "@effect/platform-bun"
import {
  HarnessIdSchema,
  type DesktopHarnessConnection,
  type HarnessConnectResult,
  type HarnessId,
} from "@magnitudedev/client-common"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { Data, Effect, Option, Schema } from "effect"
import { makeHarnessConnection } from "../server/harness-connections"
import { headlessAcnConnection } from "../server/acn-connection"
import { renderFields, renderTable, runCommand } from "./output"

class ConnectionsCommandError extends Data.TaggedError("ConnectionsCommandError")<{
  readonly message: string
}> {}

const parseHarness = (input: string) => Schema.decodeUnknown(HarnessIdSchema)(input).pipe(
  Effect.mapError(() => new ConnectionsCommandError({ message: `Unsupported harness: ${input}` })),
)

const parseModel = (input: string | undefined) => input === undefined
  ? Effect.succeed(Option.none())
  : Schema.decodeUnknown(ProviderModelIdSchema)(input).pipe(
      Effect.map(Option.some),
      Effect.mapError(() => new ConnectionsCommandError({ message: `Invalid model ID: ${input}` })),
    )

const requireRunningService = Effect.gen(function* () {
  const connection = yield* headlessAcnConnection
  yield* connection.startup.awaitReady
})

type CommandRequirements = FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor | HttpClient.HttpClient

const withService = <A>(use: (service: Effect.Effect.Success<typeof makeHarnessConnection>) => Effect.Effect<A, unknown, CommandRequirements>) =>
  Effect.scoped(Effect.gen(function* () {
    const service = yield* makeHarnessConnection
    return yield* use(service)
  })).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))

export const renderConnections = (rows: readonly DesktopHarnessConnection[]): string => {
  if (rows.length === 0) return "No supported harnesses are available.\n"
  return renderTable(rows, [
    { heading: "HARNESS", value: ({ name }) => name },
    { heading: "ID", value: ({ id }) => id },
    { heading: "INSTALLATION", value: row => row.installed ? "Installed" : "Not installed" },
    { heading: "CONNECTION", value: row => row.inspection._tag === "Unavailable" ? "Unable to check" : row.inspection._tag },
    { heading: "DETAIL", value: row => row.inspection._tag === "Connected" ? "" : row.inspection.reason },
  ])
}

export const listConnections = () => runCommand({
  effect: withService((service) => service.inspect),
  render: renderConnections,
})

export const renderAddedConnection = ({
  harness,
  model,
  connection,
}: {
  readonly harness: HarnessId
  readonly model: Option.Option<typeof ProviderModelIdSchema.Type>
  readonly connection: HarnessConnectResult
}): string => {
  const heading = `Connected ${harness} to Magnitude.`
  const fields: (readonly [string, string])[] = [
    ...(Option.isSome(model) ? [["Selected model", model.value]] as const : []),
    ...Option.match(connection.companion, {
      onNone: () => [],
      onSome: (companion) => [[companion.name, companion.status === "already-installed"
        ? "Already installed"
        : companion.status === "enabled" ? "Enabled" : "Installed"]] as const,
    }),
    ...(connection.skillInstalled ? [["Skill", "Installed"]] as const : []),
  ]
  return [
    heading,
    ...(fields.length > 0 ? [renderFields(fields)] : []),
    ...Option.match(connection.companion, {
      onNone: () => [],
      onSome: ({ activationInstructions }) => Option.match(activationInstructions, {
        onNone: () => [], onSome: (instructions) => ["", instructions],
      }),
    }),
    "",
  ].join("\n")
}

export const addConnection = (
  harnessInput: string,
  modelInput: string | undefined,
  installSkill: boolean,
) => runCommand({
  effect: withService((service) => Effect.gen(function* () {
    const harness = yield* parseHarness(harnessInput)
    const model = yield* parseModel(modelInput)
    yield* Effect.scoped(requireRunningService)
    const connection = yield* service.connect(harness, {
      model,
      installSkill,
      launchOnStartup: false,
    })
    return { harness, model, connection }
  })),
  render: renderAddedConnection,
})

export const syncConnections = (harnessInput: string | undefined) => runCommand({
  effect: withService((service) => Effect.gen(function* () {
    const harness: HarnessId | undefined = harnessInput === undefined
      ? undefined
      : yield* parseHarness(harnessInput)
    yield* Effect.scoped(requireRunningService)
    yield* service.sync(harness)
    return yield* service.inspect
  })),
  render: renderConnections,
})

export const removeConnection = (harnessInput: string) => runCommand({
  effect: withService((service) => Effect.gen(function* () {
    const harness = yield* parseHarness(harnessInput)
    yield* service.disconnect(harness)
    return harness
  })),
  render: (harness) => `Disconnected ${harness} from Magnitude.\n`,
})
