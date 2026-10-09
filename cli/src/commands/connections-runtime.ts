import {
  HarnessIdSchema,
  ProviderModelIdSchema,
  type HarnessConnectOutcome,
  type HarnessConnectionStatus,
  type HarnessId,
} from "@magnitudedev/sdk"
import { Data, Effect, Option, Schema } from "effect"
import { localHarnessConnections, requireLocalService } from "../server/local-harness-connections"
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

/** Harness connections belong to the person running the command; this process writes them into their home. */
const withConnections = <A>(use: (connections: Effect.Effect.Success<typeof localHarnessConnections>) => Effect.Effect<A, unknown>) =>
  localHarnessConnections.pipe(Effect.flatMap(use))

export const renderConnections = (rows: readonly HarnessConnectionStatus[]): string => {
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
  effect: withConnections(connections => connections.inspect),
  render: renderConnections,
})

export const renderAddedConnection = ({
  harness,
  model,
  connection,
}: {
  readonly harness: HarnessId
  readonly model: Option.Option<typeof ProviderModelIdSchema.Type>
  readonly connection: HarnessConnectOutcome
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

export const connectConnection = (
  harnessInput: string,
  modelInput: string | undefined,
  installSkill: boolean,
) => runCommand({
  effect: Effect.gen(function* () {
    const harness = yield* parseHarness(harnessInput)
    const model = yield* parseModel(modelInput)
    yield* requireLocalService
    const result = yield* withConnections(connections => connections.connect(harness, { model, installSkill, launchOnStartup: false }))
    const connection: HarnessConnectOutcome = {
      companion: Option.map(result.companion, companion => ({ name: companion.name, source: companion.source,
        securityNotice: companion.securityNotice, status: companion.status, activationInstructions: companion.activationInstructions })),
      skillInstalled: result.skillInstalled,
    }
    return { harness, model, connection }
  }),
  render: renderAddedConnection,
})

export const syncConnections = (harnessInput: string | undefined) => runCommand({
  effect: Effect.gen(function* () {
    const harness: Option.Option<HarnessId> = harnessInput === undefined ? Option.none() : Option.some(yield* parseHarness(harnessInput))
    yield* requireLocalService
    return yield* withConnections(connections => connections.sync(Option.getOrUndefined(harness)).pipe(Effect.zipRight(connections.inspect)))
  }),
  render: renderConnections,
})

export const removeConnection = (harnessInput: string) => runCommand({
  effect: Effect.gen(function* () {
    const harness = yield* parseHarness(harnessInput)
    yield* withConnections(connections => connections.disconnect(harness))
    return harness
  }),
  render: (harness) => `Disconnected ${harness} from Magnitude.\n`,
})
