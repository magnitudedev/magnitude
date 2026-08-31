import { FetchHttpClient } from "@effect/platform"
import type * as FileSystem from "@effect/platform/FileSystem"
import type * as Path from "@effect/platform/Path"
import type * as CommandExecutor from "@effect/platform/CommandExecutor"
import type * as HttpClient from "@effect/platform/HttpClient"
import { BunContext } from "@effect/platform-bun"
import {
  HarnessAvailabilitySchema,
  HarnessIdSchema,
  type HarnessConnection,
  type HarnessDestination,
  type HarnessId,
} from "@magnitudedev/client-common"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { Data, Effect, Option, Schema } from "effect"
import { makeHarnessConnection } from "../harness-connections/service"
import { existingAcnConnection } from "../server/acn-connection"
import { ensureTrailingNewline, runCommand } from "./output"

const HarnessDestinationSchema = Schema.Struct({
  id: HarnessIdSchema,
  name: Schema.String,
  availability: HarnessAvailabilitySchema,
  selectable: Schema.Boolean,
  note: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})
const ConnectionsResultSchema = Schema.Struct({
  connections: Schema.Array(HarnessDestinationSchema),
})
const HarnessLaunchPlanSchema = Schema.Struct({
  harness: HarnessIdSchema,
  executable: Schema.String,
  args: Schema.Array(Schema.String),
  environment: Schema.Record({ key: Schema.String, value: Schema.String }),
  modelId: ProviderModelIdSchema,
})
const AddConnectionResultSchema = Schema.Struct({
  action: Schema.Literal("add"),
  harness: HarnessIdSchema,
  skillInstalled: Schema.Boolean,
  launchPlan: Schema.optionalWith(HarnessLaunchPlanSchema, { as: "Option", exact: true }),
})
const RemoveConnectionResultSchema = Schema.Struct({
  action: Schema.Literal("remove"),
  harness: HarnessIdSchema,
})

class ConnectionsCommandError extends Data.TaggedError("ConnectionsCommandError")<{
  readonly code: string
  readonly message: string
  readonly retryable: boolean
}> {}

const parseHarness = (input: string) => Schema.decodeUnknown(HarnessIdSchema)(input).pipe(
  Effect.mapError(() => new ConnectionsCommandError({
    code: "unsupported_harness",
    message: `Unsupported harness: ${input}`,
    retryable: false,
  })),
)
const parseCurrentModel = (input: string | undefined) => input === undefined
  ? Effect.succeed(Option.none())
  : Schema.decodeUnknown(ProviderModelIdSchema)(input).pipe(
      Effect.map(Option.some),
      Effect.mapError(() => new ConnectionsCommandError({
        code: "invalid_model_id",
        message: `Invalid model ID: ${input}`,
        retryable: false,
      })),
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

const projectDestination = (row: HarnessDestination) => ({
  ...row,
  note: Option.fromNullable(row.note),
})
type CommandDestination = ReturnType<typeof projectDestination>

const renderRows = (rows: ReadonlyArray<CommandDestination>): string => {
  if (rows.length === 0) return "No harness connections.\n"
  const width = Math.max(...rows.map(({ name }) => name.length))
  return ensureTrailingNewline(rows.map((row) =>
    `${row.name.padEnd(width)}  ${row.availability}`
  ).join("\n"))
}

export const listConnections = () => runCommand({
  effect: withService((service) => service.list.pipe(Effect.map((connections) => ({
    connections: connections.map(projectDestination),
  })))),
  schema: ConnectionsResultSchema,
  render: ({ connections }) => renderRows(connections),
})

export const addConnection = (
  harnessInput: string,
  setCurrentInput: string | undefined,
  installSkill: boolean,
) => runCommand({
  effect: withService((service) => Effect.gen(function* () {
    yield* Effect.scoped(requireRunningService)
    const harness = yield* parseHarness(harnessInput)
    const setCurrent = yield* parseCurrentModel(setCurrentInput)
    if (installSkill) yield* service.installSkill(harness)
    const result = yield* service.connect(harness, { setCurrent })
    return { action: "add" as const, harness, skillInstalled: installSkill, launchPlan: result.launchPlan }
  })),
  schema: AddConnectionResultSchema,
  render: ({ harness, launchPlan, skillInstalled }) => Option.match(launchPlan, {
    onNone: () => `Connected all Magnitude models to ${harness}.${skillInstalled ? " Installed the Magnitude skill." : ""}\n`,
    onSome: (plan) => [
      `Connected all Magnitude models to ${harness}.`,
      ...(skillInstalled ? ["Installed the Magnitude skill."] : []),
      `Current model: ${plan.modelId}`,
      `Launch: ${[plan.executable, ...plan.args].join(" ")}`,
      "",
    ].join("\n"),
  }),
})

export const syncConnections = (harnessInput: string | undefined) => runCommand({
  effect: withService((service) => Effect.gen(function* () {
    yield* Effect.scoped(requireRunningService)
    const harness: HarnessId | undefined = harnessInput === undefined
      ? undefined
      : yield* parseHarness(harnessInput)
    const connections = yield* service.sync(harness)
    return { connections: connections.map(projectDestination) }
  })),
  schema: ConnectionsResultSchema,
  render: ({ connections }) => renderRows(connections),
})

export const removeConnection = (harnessInput: string) => runCommand({
  effect: withService((service) => Effect.gen(function* () {
    const harness = yield* parseHarness(harnessInput)
    yield* service.disconnect(harness)
    return { action: "remove" as const, harness }
  })),
  schema: RemoveConnectionResultSchema,
  render: ({ harness }) => `Removed ${harness}.\n`,
})
