import * as Command from "@effect/platform/Command"
import { FetchHttpClient } from "@effect/platform"
import * as FileSystem from "@effect/platform/FileSystem"
import * as BunContext from "@effect/platform-bun/BunContext"
import * as BunRuntime from "@effect/platform-bun/BunRuntime"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import {
  interactiveProcessExitCode,
  runInteractiveProcess,
  type InteractiveProcessTermination,
} from "@magnitudedev/utils/process"
import { Console, Effect, Option, Schema } from "effect"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { buildLocalIcn } from "../inference/scripts/build-local"
import { makeAcnConnectionWithInstanceManager } from "../cli/src/server/acn-connection"
import {
  makeBootstrappingAcnInstanceManager,
  stopLocalAcn,
} from "../cli/src/server/acn-instance-manager"
import {
  serviceStatus,
  startInstalledService,
  stopService,
} from "../cli/src/server/service"

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const cliEntrypoint = resolve(projectRoot, "cli/src/index.tsx")
const acnEntrypoint = resolve(projectRoot, "packages/acn/src/binary.ts")
const piPackageSource = resolve(projectRoot, "integrations/pi")
const encodeJsonString = Schema.encodeSync(Schema.parseJson(Schema.String))

const StatusSuccessEnvelopeSchema = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  command: Schema.Literal("models.status"),
  ok: Schema.Literal(true),
  data: Schema.Union(
    Schema.Struct({ state: Schema.Literal("initializing"), models: Schema.Tuple() }),
    Schema.Struct({
      state: Schema.Literal("ready"),
      models: Schema.Array(Schema.Struct({
        modelId: ProviderModelIdSchema,
        installation: Schema.Literal("not_installed", "installing", "installed", "removing", "unavailable"),
      })),
    }),
  ),
})

const StatusFailureEnvelopeSchema = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  command: Schema.Literal("models.status"),
  ok: Schema.Literal(false),
  error: Schema.Struct({ message: Schema.NonEmptyString }),
})

const StatusEnvelopeSchema = Schema.parseJson(Schema.Union(
  StatusSuccessEnvelopeSchema,
  StatusFailureEnvelopeSchema,
))

class PiDevelopmentFailed extends Schema.TaggedError<PiDevelopmentFailed>()(
  "PiDevelopmentFailed",
  { message: Schema.String },
) {}

const requireSuccess = (operation: string, termination: InteractiveProcessTermination) => {
  const exitCode = interactiveProcessExitCode(termination)
  return exitCode === 0
    ? Effect.void
    : Effect.fail(new PiDevelopmentFailed({ message: `${operation} exited with status ${exitCode}` }))
}

export const decodePiDevelopmentStatus = (
  source: string,
): Effect.Effect<typeof StatusSuccessEnvelopeSchema.Type, PiDevelopmentFailed> => Schema.decodeUnknown(StatusEnvelopeSchema)(source).pipe(
  Effect.mapError(() => new PiDevelopmentFailed({
    message: "The development CLI returned an incompatible model status response",
  })),
  Effect.flatMap((status) => {
    if (!status.ok) return Effect.fail(new PiDevelopmentFailed({ message: status.error.message }))
    return Effect.succeed(status)
  }),
)

const buildDevelopmentIcn = Effect.tryPromise({
  try: () => buildLocalIcn({ diagnostics: "errors" }),
  catch: (error) => new PiDevelopmentFailed({
    message: `Could not build the development inference runtime: ${String(error)}`,
  }),
})

const program = Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const temporaryDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-dev-" })
  const magnitudeExecutable = resolve(temporaryDirectory, "magnitude")
  yield* fs.writeFileString(
    magnitudeExecutable,
    `#!/usr/bin/env bun\nimport ${encodeJsonString(cliEntrypoint)}\n`,
  )
  yield* fs.chmod(magnitudeExecutable, 0o755)

  yield* Console.log("Preparing the local inference runtime...")
  const localIcn = yield* buildDevelopmentIcn
  process.env.MAGNITUDE_ICN_PATH = localIcn.installationPath

  const previousService = yield* serviceStatus
  if (previousService.running && !previousService.managed) {
    return yield* new PiDevelopmentFailed({
      message: "A Magnitude service not owned by the installed service manager is already running; stop it before launching Pi development",
    })
  }
  yield* Effect.addFinalizer(() => stopLocalAcn.pipe(
    Effect.ignore,
    Effect.zipRight(previousService.running && previousService.managed
      ? startInstalledService.pipe(Effect.catchAll((error) =>
          Effect.logError("Could not restore the installed Magnitude service", error)))
      : Effect.void),
  ))
  yield* stopService

  yield* Console.log(`Starting the ${localIcn.backend} development runtime...`)
  const manager = yield* makeBootstrappingAcnInstanceManager({
    launchCommand: Option.some(["bun", acnEntrypoint, "serve"]),
    debug: false,
  })
  const acnConnection = yield* makeAcnConnectionWithInstanceManager(manager)
  yield* acnConnection.startup.awaitReady.pipe(
    Effect.mapError((error) => new PiDevelopmentFailed({
      message: `Could not start the development Magnitude service: ${error.message}`,
    })),
  )

  let models: ReadonlyArray<{ readonly modelId: typeof ProviderModelIdSchema.Type; readonly installation: string }> = []
  let statusReady = false
  for (let attempt = 0; attempt < 60; attempt += 1) {
    const statusSource = yield* Command.string(Command.make(
      magnitudeExecutable,
      "models",
      "status",
      "--json",
    ))
    const status = yield* decodePiDevelopmentStatus(statusSource)
    if (status.data.state === "ready") {
      models = status.data.models
      statusReady = true
      break
    }
    yield* Effect.sleep("500 millis")
  }
  if (!statusReady) {
    return yield* new PiDevelopmentFailed({
      message: "Magnitude model discovery did not become ready within 30 seconds",
    })
  }
  const model = models.find(({ installation }) => installation === "installed")
  if (model === undefined) {
    return yield* new PiDevelopmentFailed({
      message: "Install a Magnitude model before starting Pi development",
    })
  }

  yield* Console.log(`Connecting the local Pi package with ${model.modelId}...`)
  const harnessConnection = yield* runInteractiveProcess({
    executable: magnitudeExecutable,
    args: ["connections", "add", "pi", "--set-model", model.modelId],
    environment: {
      ...process.env,
      MAGNITUDE_PI_PACKAGE_SOURCE: piPackageSource,
    },
    workingDirectory: projectRoot,
  })
  yield* requireSuccess("Magnitude Pi connection", harnessConnection)

  yield* Console.log("Launching Pi with the local Magnitude CLI and extension...")
  const pi = yield* runInteractiveProcess({
    executable: "pi",
    args: ["--model", `magnitude/${model.modelId}`],
    environment: {
      ...process.env,
      MAGNITUDE_CLI: magnitudeExecutable,
    },
    workingDirectory: projectRoot,
  })
  yield* requireSuccess("Pi", pi)
}))

if (import.meta.main) {
  BunRuntime.runMain(program.pipe(Effect.provide([
    BunContext.layer,
    FetchHttpClient.layer,
  ])))
}
