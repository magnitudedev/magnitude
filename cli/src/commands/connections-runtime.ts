import {
  HarnessIdSchema,
  ProviderModelIdSchema,
  type HarnessConnectOutcome,
  type HarnessConnectionStatus,
  type HarnessId,
  type MagnitudeClient,
} from "@magnitudedev/sdk"
import { Data, Effect, Option, Schema, Stream } from "effect"
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

/** Harness connections belong to the running service, which configures harnesses on its own machine. */
const withClient = <A>(use: (client: Pick<MagnitudeClient, "connections">) => Effect.Effect<A, unknown>) =>
  Effect.scoped(Effect.gen(function* () {
    const connection = yield* existingAcnConnection
    yield* connection.startup.awaitReady
    return yield* use(connection.client)
  }))

const readConnections = (client: Pick<MagnitudeClient, "connections">) => client.connections.watchHarnessConnections({}).pipe(
  Stream.runHead,
  Effect.flatMap(Option.match({
    onNone: () => Effect.fail(new ConnectionsCommandError({ message: "Magnitude did not report harness connections." })),
    onSome: snapshot => snapshot._tag === "Ready" ? Effect.succeed(snapshot.connections) : Effect.fail(new ConnectionsCommandError({ message: snapshot.message })),
  })),
)

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
  effect: withClient(readConnections),
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

export const addConnection = (
  harnessInput: string,
  modelInput: string | undefined,
  installSkill: boolean,
) => runCommand({
  effect: Effect.gen(function* () {
    const harness = yield* parseHarness(harnessInput)
    const model = yield* parseModel(modelInput)
    const connection = yield* withClient(client => client.connections.connectHarness({ harness, model, installSkill }))
    return { harness, model, connection }
  }),
  render: renderAddedConnection,
})

export const syncConnections = (harnessInput: string | undefined) => runCommand({
  effect: Effect.gen(function* () {
    const harness: Option.Option<HarnessId> = harnessInput === undefined ? Option.none() : Option.some(yield* parseHarness(harnessInput))
    return yield* withClient(client => client.connections.syncHarnessConnections({ harness }).pipe(Effect.zipRight(readConnections(client))))
  }),
  render: renderConnections,
})

export const removeConnection = (harnessInput: string) => runCommand({
  effect: Effect.gen(function* () {
    const harness = yield* parseHarness(harnessInput)
    yield* withClient(client => client.connections.disconnectHarness({ harness }))
    return harness
  }),
  render: (harness) => `Disconnected ${harness} from Magnitude.\n`,
})
