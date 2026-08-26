import type { Command as Commander } from "@commander-js/extra-typings"
import { BunContext } from "@effect/platform-bun"
import { FetchHttpClient } from "@effect/platform"
import type * as FileSystem from "@effect/platform/FileSystem"
import type * as Path from "@effect/platform/Path"
import type * as CommandExecutor from "@effect/platform/CommandExecutor"
import type * as HttpClient from "@effect/platform/HttpClient"
import {
  HarnessIdSchema,
  type HarnessConnection,
  type HarnessDestination,
  type HarnessId,
} from "@magnitudedev/client-common"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { Data, Effect, Option, Schema } from "effect"
import { makeHarnessConnection } from "../harness-connections/service"
import { startServer } from "./server"

class ConnectionsCommandError extends Data.TaggedError("ConnectionsCommandError")<{
  readonly message: string
}> {}

const parseHarness = (input: string) => Schema.decodeUnknown(HarnessIdSchema)(input).pipe(
  Effect.mapError(() => new ConnectionsCommandError({ message: `Unsupported harness: ${input}` })),
)

const parseCurrentModel = (input: string | undefined) => input === undefined
  ? Effect.succeed(Option.none())
  : Schema.decodeUnknown(ProviderModelIdSchema)(input).pipe(
      Effect.map(Option.some),
      Effect.mapError(() => new ConnectionsCommandError({ message: `Invalid model ID: ${input}` })),
    )

const printRows = (rows: ReadonlyArray<HarnessDestination>): void => {
  const width = Math.max(...rows.map(({ name }) => name.length))
  for (const row of rows) {
    process.stdout.write(`${row.name.padEnd(width)}  ${row.availability}\n`)
  }
}

// Kept local so Commander actions all acquire the same concrete implementation
// shape as onboarding without creating command-specific adapter paths.
type CommandRequirements = FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor | HttpClient.HttpClient

const execute = <A>(use: (service: HarnessConnection) => Effect.Effect<A, unknown, CommandRequirements>) =>
  Effect.runPromise(Effect.gen(function* () {
    const service = yield* makeHarnessConnection
    return yield* use(service)
  }).pipe(
    Effect.provide([BunContext.layer, FetchHttpClient.layer]),
    Effect.catchAll((error) => Effect.sync(() => {
      const message = error instanceof Error && "message" in error ? error.message : String(error)
      process.stderr.write(`${message}\n`)
      process.exitCode = 1
    })),
  ))

export const registerConnectionsCommand = (program: Commander): void => {
  const connections = program.command("connections")
    .description("Manage Magnitude harness connections")

  connections.command("list")
    .description("List supported harnesses and installation status")
    .action(() => execute((service) => service.list.pipe(
      Effect.tap((rows) => Effect.sync(() => printRows(rows))),
      Effect.asVoid,
    )))

  connections.command("add")
    .description("Connect every installed Magnitude model to a harness")
    .argument("<harness>", "Harness ID")
    .option("--set-current <model-id>", "Also select this Magnitude model in the harness")
    .action((harnessInput, options) => execute((service) => Effect.gen(function* () {
      yield* startServer
      const harness = yield* parseHarness(harnessInput)
      const setCurrent = yield* parseCurrentModel(options.setCurrent)
      const result = yield* service.connect(harness, { setCurrent })
      yield* Effect.sync(() => {
        process.stdout.write(`Connected all Magnitude models to ${harness}.\n`)
        if (Option.isSome(result.launchPlan)) {
          const plan = result.launchPlan.value
          process.stdout.write(`Current model: ${plan.modelId}\nLaunch: ${[plan.executable, ...plan.args].join(" ")}\n`)
        }
      })
    })))

  connections.command("sync")
    .description("Reconcile configured harnesses with their Magnitude bindings")
    .argument("[harness]", "Harness ID")
    .action((harnessInput) => execute((service) => Effect.gen(function* () {
      yield* startServer
      const harness: HarnessId | undefined = harnessInput === undefined
        ? undefined
        : yield* parseHarness(harnessInput)
      const rows = yield* service.sync(harness)
      yield* Effect.sync(() => printRows(rows))
    })))

  connections.command("remove")
    .description("Remove a Magnitude harness connection")
    .argument("<harness>", "Harness ID")
    .action((harnessInput) => execute((service) => Effect.gen(function* () {
      const harness = yield* parseHarness(harnessInput)
      yield* service.disconnect(harness)
      yield* Effect.sync(() => process.stdout.write(`Removed ${harness}.\n`))
    })))
}
