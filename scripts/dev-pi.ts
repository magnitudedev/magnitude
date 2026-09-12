import * as Command from "@effect/platform/Command"
import { FetchHttpClient } from "@effect/platform"
import * as FileSystem from "@effect/platform/FileSystem"
import * as BunContext from "@effect/platform-bun/BunContext"
import * as BunRuntime from "@effect/platform-bun/BunRuntime"
import { ProviderModelIdSchema, localModelIsInstalled, type ModelCatalogState, type LocalModel, formatConnectionError } from "@magnitudedev/sdk"
import { HarnessIdSchema } from "@magnitudedev/client-common"
import { harnessExecutableSearchPath, makeHarnessConnectionService } from "@magnitudedev/harness-connections"
import { piDevelopmentConnectionOptions } from "../cli/src/server/harness-connections"
import {
  interactiveProcessExitCode,
  runInteractiveProcess,
  type InteractiveProcessTermination,
} from "@magnitudedev/utils/process"
import { Console, Effect, Option, Schema } from "effect"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { buildLocalIcn } from "../inference/scripts/build-local"
import { headlessAcnConnection } from "../cli/src/server/acn-connection"
import { desktopServiceOrigin } from "../cli/src/server/application"
import { BunSqliteDriverLayer } from "@magnitudedev/daemon-management/bun"

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const cliEntrypoint = resolve(projectRoot, "cli/src/index.ts")
const piPackageSource = resolve(projectRoot, "integrations/pi")
const encodeJsonString = Schema.encodeSync(Schema.parseJson(Schema.String))

export const piDevelopmentArgs = (modelId: string | undefined, skillFile: string): string[] => [
  ...(modelId === undefined ? [] : ["--model", `magnitude/${modelId}`]),
  // Pi also discovers ~/.agents/skills outside PI_CODING_AGENT_DIR. Explicit
  // skills remain enabled with --no-skills; ambient discovery must not win.
  "--no-skills", "--skill", skillFile,
]

class PiDevelopmentFailed extends Schema.TaggedError<PiDevelopmentFailed>()(
  "PiDevelopmentFailed",
  { message: Schema.String },
) {}

export const resolvePiDevelopmentExecutable = (searchPath = process.env.PATH ?? "") =>
  Effect.sync(() => Bun.which("pi", { PATH: harnessExecutableSearchPath(searchPath) })).pipe(
    Effect.flatMap((executable) => executable === null
      ? Effect.fail(new PiDevelopmentFailed({ message: "Pi is not installed on your PATH" }))
      : Effect.succeed(executable)),
  )

const requireSuccess = (operation: string, termination: InteractiveProcessTermination) => {
  const exitCode = interactiveProcessExitCode(termination)
  return exitCode === 0
    ? Effect.void
    : Effect.fail(new PiDevelopmentFailed({ message: `${operation} exited with status ${exitCode}` }))
}

const buildDevelopmentIcn = Effect.tryPromise({
  try: () => buildLocalIcn({ diagnostics: "errors" }),
  catch: (error) => new PiDevelopmentFailed({
    message: `Could not build the development inference runtime: ${String(error)}`,
  }),
})

export const awaitPiDevelopmentModel = <E, R>(
  read: Effect.Effect<ModelCatalogState, E, R>,
) => {
  const poll: Effect.Effect<LocalModel, E, R> = Effect.suspend(() => read.pipe(
    Effect.flatMap((status) => {
      // A ready snapshot can still be empty while startup discovery runs.
      const model = status._tag === "Initializing" ? undefined : status.models.flatMap(entry => entry._tag === "Local" ? [entry.product] : []).find(localModelIsInstalled)
      return model === undefined
        ? Effect.sleep("500 millis").pipe(Effect.zipRight(poll))
        : Effect.succeed(model)
    }),
  ))
  return poll.pipe(Effect.timeoutFail({
    duration: "30 seconds",
    onTimeout: () => new PiDevelopmentFailed({ message: "No installed Magnitude model became available within 30 seconds. Check `magnitude models status` or install a model before starting Pi development." }),
  }))
}

const program = Effect.scoped(Effect.gen(function* () {
  const args = process.argv.slice(2)
  if (args.some(arg => arg !== "--setup")) {
    return yield* new PiDevelopmentFailed({ message: "Usage: bun run dev:pi [--setup]" })
  }
  const freshSetup = args.includes("--setup")
  const piExecutable = yield* resolvePiDevelopmentExecutable()
  const fs = yield* FileSystem.FileSystem
  const temporaryDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-dev-" })
  const magnitudeExecutable = resolve(temporaryDirectory, "magnitude")
  yield* fs.writeFileString(
    magnitudeExecutable,
    `#!/usr/bin/env bun\nimport ${encodeJsonString(cliEntrypoint)}\n`,
  )
  yield* fs.chmod(magnitudeExecutable, 0o755)

  const piDirectory = resolve(temporaryDirectory, "pi")
  const connectionOptions = piDevelopmentConnectionOptions(temporaryDirectory)
  const { paths } = connectionOptions
  yield* Command.make("bun", "run", "build").pipe(Command.workingDirectory(piPackageSource), Command.exitCode,
    Effect.flatMap((code) => code === 0 ? Effect.void : Effect.fail(new PiDevelopmentFailed({ message: "Could not build the local Pi extension" }))))

  yield* Console.log("Preparing the local inference runtime...")
  const localIcn = yield* buildDevelopmentIcn
  const previousIcnPath = process.env.MAGNITUDE_ICN_PATH
  yield* Effect.addFinalizer(() => Effect.sync(() => {
    if (previousIcnPath === undefined) delete process.env.MAGNITUDE_ICN_PATH
    else process.env.MAGNITUDE_ICN_PATH = previousIcnPath
  }))
  process.env.MAGNITUDE_ICN_PATH = localIcn.installationPath

  yield* Command.make("bun", "run", "build").pipe(Command.workingDirectory(resolve(projectRoot, "desktop")), Command.exitCode,
    Effect.flatMap(code => code === 0 ? Effect.void : Effect.fail(new PiDevelopmentFailed({ message: "Could not build the development desktop" }))))
  yield* Console.log("Ensuring the Magnitude development app is running in the background...")
  const acnConnection = yield* headlessAcnConnection
  yield* acnConnection.startup.awaitReady.pipe(
    Effect.mapError((error) => new PiDevelopmentFailed({
      message: `Could not start the development Magnitude service: ${formatConnectionError(error)}`,
    })),
  )

  let modelId: string | undefined
  if (freshSetup) {
    yield* Console.log("Installing only the local Pi package into a fresh temporary profile...")
    const code = yield* Command.make(piExecutable, "install", piPackageSource).pipe(
      Command.env({ PI_CODING_AGENT_DIR: piDirectory }),
      Command.exitCode,
    )
    if (code !== 0) return yield* new PiDevelopmentFailed({ message: "Could not install the local Pi package" })
  } else {
    yield* Console.log("Waiting for an installed Magnitude model...")
    const model = yield* awaitPiDevelopmentModel(acnConnection.client.models.getCatalog({}))
    modelId = model.modelId
    yield* Console.log(`Connecting the local Pi package with ${model.modelId}...`)
    const connection = yield* makeHarnessConnectionService(connectionOptions)
    yield* connection.connect(HarnessIdSchema.make("pi"), { model: Option.some(ProviderModelIdSchema.make(model.modelId)) })
  }

  yield* Console.log(`Launching ${piExecutable} with the local Magnitude CLI and extension...`)
  const pi = yield* runInteractiveProcess({
    executable: piExecutable,
    args: piDevelopmentArgs(modelId, freshSetup
      ? resolve(piPackageSource, "dist/skills/magnitude/SKILL.md")
      : paths.skillInstallations["shared-agents"].skillFile),
    environment: {
      ...process.env,
      MAGNITUDE_CLI: magnitudeExecutable,
      PI_CODING_AGENT_DIR: piDirectory,
      MAGNITUDE_PI_DEVELOPMENT_ROOT: temporaryDirectory,
      MAGNITUDE_PI_DEVELOPMENT_ORIGIN: desktopServiceOrigin,
    },
  })
  yield* requireSuccess("Pi", pi)
}))

if (import.meta.main) {
  BunRuntime.runMain(program.pipe(Effect.provide([
    BunContext.layer,
    FetchHttpClient.layer,
    BunSqliteDriverLayer,
  ])))
}
