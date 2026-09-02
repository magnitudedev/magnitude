import { FetchHttpClient } from "@effect/platform"
import type * as CommandExecutor from "@effect/platform/CommandExecutor"
import type * as FileSystem from "@effect/platform/FileSystem"
import type * as HttpClient from "@effect/platform/HttpClient"
import type * as Path from "@effect/platform/Path"
import { BunContext } from "@effect/platform-bun"
import {
  HarnessIdSchema,
  type HarnessConnection,
  type HarnessDestination,
  type HarnessId,
  type HarnessLaunchPlan,
} from "@magnitudedev/client-common"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { Data, Effect, Option, Schema } from "effect"
import { makeHarnessConnection } from "../harness-connections/service"
import { existingAcnConnection } from "../server/acn-connection"
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
  const connection = yield* existingAcnConnection
  yield* connection.startup.awaitReady
})

type CommandRequirements = FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor | HttpClient.HttpClient

const withService = <A>(use: (service: HarnessConnection) => Effect.Effect<A, unknown, CommandRequirements>) =>
  Effect.scoped(Effect.gen(function* () {
    const service = yield* makeHarnessConnection
    return yield* use(service)
  })).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))

const connectionStatus = (row: HarnessDestination): string => {
  if (row.id === "magnitude") return "Built in"
  if (row.connected) return "Connected"
  return row.availability === "Installed" ? "Available" : "Not installed"
}

export const renderConnections = (rows: readonly HarnessDestination[]): string => {
  if (rows.length === 0) return "No supported harnesses are available.\n"
  return renderTable(rows, [
    { heading: "HARNESS", value: ({ name }) => name },
    { heading: "ID", value: ({ id }) => id },
    { heading: "STATUS", value: connectionStatus },
  ])
}

export const listConnections = () => runCommand({
  effect: withService((service) => service.list),
  render: renderConnections,
})

const posixQuote = (value: string): string => /^[A-Za-z0-9_./:@%+=,-]+$/.test(value)
  ? value
  : `'${value.replaceAll("'", `'"'"'`)}'`

export const renderLaunchPlan = (plan: HarnessLaunchPlan): string => {
  if (process.platform === "win32") {
    const quote = (value: string) => `"${value.replaceAll('"', '`"')}"`
    const environment = Object.entries(plan.environment)
      .map(([key, value]) => `$env:${key} = ${quote(value)}`)
    return [...environment, `& ${[plan.command, ...plan.args].map(quote).join(" ")}`].join("\n  ")
  }
  const environment = Object.entries(plan.environment)
    .map(([key, value]) => `${key}=${posixQuote(value)}`)
  return [...environment, posixQuote(plan.command), ...plan.args.map(posixQuote)].join(" ")
}

export const addConnection = (
  harnessInput: string,
  modelInput: string | undefined,
  installSkill: boolean,
) => runCommand({
  effect: withService((service) => Effect.gen(function* () {
    yield* Effect.scoped(requireRunningService)
    const harness = yield* parseHarness(harnessInput)
    const model = yield* parseModel(modelInput)
    if (installSkill) yield* service.installSkill(harness)
    yield* service.connect(harness, { model })
    const launchPlan = yield* Option.match(model, {
      onNone: () => Effect.succeed(Option.none()),
      onSome: (modelId) => service.launch(harness, modelId).pipe(Effect.map(Option.some)),
    })
    return { harness, model, skillInstalled: installSkill, launchPlan }
  })),
  render: ({ harness, model, skillInstalled, launchPlan }) => {
    const heading = harness === "magnitude"
      ? "Magnitude Harness is built in."
      : `Connected ${harness} to Magnitude.`
    const fields: (readonly [string, string])[] = [
      ...(Option.isSome(model) ? [["Selected model", model.value]] as const : []),
      ...(skillInstalled ? [["Skill", "Installed"]] as const : []),
    ]
    return [
      heading,
      ...(fields.length > 0 ? [renderFields(fields)] : []),
      ...Option.match(launchPlan, {
        onNone: () => [],
        onSome: (plan) => ["", `Open ${harness} with this model:`, `  ${renderLaunchPlan(plan)}`],
      }),
      "",
    ].join("\n")
  },
})

export const syncConnections = (harnessInput: string | undefined) => runCommand({
  effect: withService((service) => Effect.gen(function* () {
    yield* Effect.scoped(requireRunningService)
    const harness: HarnessId | undefined = harnessInput === undefined
      ? undefined
      : yield* parseHarness(harnessInput)
    return yield* service.sync(harness)
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
