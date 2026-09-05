import * as Command from "@effect/platform/Command"
import { FetchHttpClient } from "@effect/platform"
import * as FileSystem from "@effect/platform/FileSystem"
import * as BunContext from "@effect/platform-bun/BunContext"
import * as BunRuntime from "@effect/platform-bun/BunRuntime"
import { ProviderModelIdSchema, localModelIsInstalled, type ModelCatalogState, type LocalModel, formatConnectionError } from "@magnitudedev/sdk"
import { HarnessIdSchema } from "@magnitudedev/client-common"
import { harnessConnectionPaths, makeHarnessConnectionService, makeHarnessConnectorRegistry } from "../cli/src/harness-connections/service"
import {
  interactiveProcessExitCode,
  runInteractiveProcess,
  type InteractiveProcessTermination,
} from "@magnitudedev/utils/process"
import { Cause, Console, Effect, Exit, Fiber, Option, Schema, Stream } from "effect"
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

export const piDevelopmentArgs = (modelId: string, skillFile: string): string[] => [
  "--model", `magnitude/${modelId}`,
  // Pi also discovers ~/.agents/skills outside PI_CODING_AGENT_DIR. Explicit
  // skills remain enabled with --no-skills; ambient discovery must not win.
  "--no-skills", "--skill", skillFile,
]

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
  const fs = yield* FileSystem.FileSystem
  const temporaryDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-dev-" })
  const magnitudeExecutable = resolve(temporaryDirectory, "magnitude")
  yield* fs.writeFileString(
    magnitudeExecutable,
    `#!/usr/bin/env bun\nimport ${encodeJsonString(cliEntrypoint)}\n`,
  )
  yield* fs.chmod(magnitudeExecutable, 0o755)

  const piDirectory = resolve(temporaryDirectory, "pi")
  const defaults = harnessConnectionPaths()
  const paths = {
    ...defaults,
    manifest: resolve(temporaryDirectory, "connections.json"),
    piModels: resolve(piDirectory, "models.json"),
    piSettings: resolve(piDirectory, "settings.json"),
    skillInstallations: { ...defaults.skillInstallations, "shared-agents": { skillFile: resolve(temporaryDirectory, "skills/magnitude/SKILL.md") } },
  }
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

  const previousService = yield* serviceStatus
  if (previousService.running && !previousService.managed) {
    return yield* new PiDevelopmentFailed({
      message: "A Magnitude service not owned by the installed service manager is already running; stop it before launching Pi development",
    })
  }
  yield* Effect.addFinalizer(() => Effect.gen(function* () {
    const stopped = yield* stopLocalAcn.pipe(Effect.interruptible, Effect.fork, Effect.flatMap(Fiber.await))
    if (previousIcnPath === undefined) delete process.env.MAGNITUDE_ICN_PATH
    else process.env.MAGNITUDE_ICN_PATH = previousIcnPath
    const restored = yield* (previousService.running && previousService.managed ? startInstalledService : Effect.void).pipe(Effect.interruptible, Effect.fork, Effect.flatMap(Fiber.await))
    const failures: string[] = []
    if (Exit.isFailure(stopped)) failures.push(Cause.pretty(stopped.cause))
    if (Exit.isFailure(restored)) failures.push(Cause.pretty(restored.cause))
    if (failures.length) return yield* Effect.die(new PiDevelopmentFailed({ message: `Could not restore the installed service: ${failures.join("; ")}` }))
  }))
  yield* stopService

  yield* Console.log(`Starting the ${localIcn.backend} development runtime...`)
  const manager = yield* makeBootstrappingAcnInstanceManager({
    launchCommand: Option.some(["bun", acnEntrypoint, "serve"]),
    debug: false,
  })
  const acnConnection = yield* makeAcnConnectionWithInstanceManager(manager)
  yield* acnConnection.startup.awaitReady.pipe(
    Effect.mapError((error) => new PiDevelopmentFailed({
      message: `Could not start the development Magnitude service: ${formatConnectionError(error)}`,
    })),
  )

  yield* Console.log("Waiting for an installed Magnitude model...")
  const model = yield* awaitPiDevelopmentModel(acnConnection.client.models.getCatalog({}))
  yield* Console.log(`Connecting the local Pi package with ${model.modelId}...`)
  const connection = yield* makeHarnessConnectionService({
    paths,
    registry: makeHarnessConnectorRegistry(paths, { piCompanionSource: piPackageSource }),
  })
  yield* connection.connect(HarnessIdSchema.make("pi"), { model: Option.some(ProviderModelIdSchema.make(model.modelId)) })

  yield* Console.log("Launching Pi with the local Magnitude CLI and extension...")
  const pi = yield* runInteractiveProcess({
    executable: "pi",
    args: piDevelopmentArgs(model.modelId, paths.skillInstallations["shared-agents"].skillFile),
    environment: {
      ...process.env,
      MAGNITUDE_CLI: magnitudeExecutable,
      PI_CODING_AGENT_DIR: piDirectory,
    },
  })
  yield* requireSuccess("Pi", pi)
}))

if (import.meta.main) {
  BunRuntime.runMain(program.pipe(Effect.provide([
    BunContext.layer,
    FetchHttpClient.layer,
  ])))
}
