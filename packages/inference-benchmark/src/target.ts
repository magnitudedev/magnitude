import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import { makeIcnApiClient } from "@magnitudedev/icn-protocol/client"
import { Chunk, Context, Data, Effect, Layer, Option, Ref, Schedule, Scope, Stream } from "effect"
import { homedir } from "node:os"
import { dirname, join, resolve } from "node:path"
import type {
  ExistingTarget,
  ManagedTarget,
  ModelIdentity,
  TargetConfiguration,
} from "./domain"
import {
  EndpointClient,
  type EndpointConfiguration,
  probeEndpoint,
} from "./transport"

export class TargetError extends Data.TaggedError("TargetError")<{
  readonly targetId: string
  readonly operation: string
  readonly message: string
}> {}

export interface TargetSession {
  readonly target: TargetConfiguration
  readonly endpoint: EndpointConfiguration
  readonly rootPid: Option.Option<number>
  readonly diagnostic: Effect.Effect<string>
}

const targetError = (targetId: string, operation: string, error: unknown) =>
  new TargetError({
    targetId,
    operation,
    message: error instanceof Error ? error.message : String(error),
  })

const appendBounded = (output: Ref.Ref<string>, chunk: string, limit = 1_000_000) =>
  Ref.update(output, (current) => {
    const next = `${current}${chunk}`
    return next.length <= limit ? next : next.slice(next.length - limit)
  })

const stopProcess = (
  target: ManagedTarget,
  processHandle: CommandExecutor.Process,
): Effect.Effect<void> =>
  Effect.gen(function* () {
    if (!(yield* processHandle.isRunning)) return
    yield* processHandle.kill("SIGINT").pipe(Effect.ignore)
    const stopped = yield* processHandle.isRunning.pipe(
      Effect.flatMap((running) => running ? Effect.fail(undefined) : Effect.succeed(true)),
      Effect.retry(Schedule.spaced("100 millis").pipe(Schedule.compose(Schedule.recurs(99)))),
      Effect.catchAll(() => Effect.succeed(false)),
    )
    if (stopped) return
    yield* processHandle.kill("SIGKILL").pipe(Effect.ignore)
  }).pipe(
    Effect.catchAll((error) =>
      Effect.logWarning(`Failed to stop managed target ${target.id}`).pipe(
        Effect.annotateLogs({ error: String(error) }),
      )),
  )

const waitForReady = (
  target: ManagedTarget,
  processHandle: CommandExecutor.Process,
  endpoint: EndpointConfiguration,
): Effect.Effect<void, TargetError, EndpointClient> => {
  const attempt = Effect.gen(function* () {
    if (!(yield* processHandle.isRunning.pipe(
      Effect.mapError((error) => targetError(target.id, "readiness", error)),
    ))) {
      const code = yield* processHandle.exitCode.pipe(
        Effect.mapError((error) => targetError(target.id, "readiness", error)),
      )
      return yield* new TargetError({
        targetId: target.id,
        operation: "process-exited",
        message: `process exited with ${code}`,
      })
    }
    yield* probeEndpoint(endpoint, target.readinessPath).pipe(
      Effect.mapError((error) => targetError(target.id, "readiness-pending", error)),
    )
  })
  return attempt.pipe(
    Effect.retry({
      schedule: Schedule.spaced("250 millis"),
      while: (error) => error.operation === "readiness-pending",
    }),
    Effect.timeoutFail({
      duration: "10 minutes",
      onTimeout: () => new TargetError({
        targetId: target.id,
        operation: "readiness",
        message: "target did not become ready within 10 minutes",
      }),
    }),
  )
}

const provisionIcnModel = (
  target: ManagedTarget,
): Effect.Effect<{
  readonly servedModel: string
  readonly instanceId: string
  readonly parallelSequences: number
}, TargetError, HttpClient.HttpClient | Scope.Scope> =>
  Effect.gen(function* () {
    const load = target.modelLoad
    if (target.engine !== "icn" || load === undefined) {
      return yield* new TargetError({
        targetId: target.id,
        operation: "provision",
        message: "managed ICN target is missing its model-load specification",
      })
    }

    const client = yield* makeIcnApiClient({ baseUrl: new URL(target.endpoint) })
    const readInventory: Effect.Effect<
      Awaited<ReturnType<typeof client.models.listInstalledModels> extends Effect.Effect<infer A, any, any> ? A : never>,
      TargetError
    > = Effect.suspend(() => client.models.listInstalledModels({}).pipe(
      Effect.mapError((error) => targetError(target.id, "discover-model", error)),
      Effect.flatMap((snapshot) => snapshot.reconciliationComplete
        ? Effect.succeed(snapshot)
        : Effect.sleep("250 millis").pipe(Effect.zipRight(readInventory))),
    ))
    const snapshot = yield* readInventory.pipe(
      Effect.timeoutFail({
        duration: "10 minutes",
        onTimeout: () => new TargetError({
          targetId: target.id,
          operation: "discover-model",
          message: "ICN model inventory did not finish reconciliation",
        }),
      }),
    )
    const installed = snapshot.packages.find(({ package: modelPackage }) =>
      modelPackage.files.some(({ sha256 }) => sha256 === load.artifactSha256))
    if (!installed) {
      return yield* new TargetError({
        targetId: target.id,
        operation: "discover-model",
        message: `ICN has no installed package containing artifact ${load.artifactSha256}`,
      })
    }

    const assessment = yield* client.models.assessModels({
      payload: {
        requests: [{
          requestId: `benchmark-${target.id}`,
          bundle: {
            _tag: "Standalone",
            package: { _tag: "Installed", packageId: installed.package.id },
          },
          profiles: [{
            profile: { contextLength: load.contextLimit },
            performanceContextTokens: [load.contextLimit],
          }],
        }],
      },
    }).pipe(Effect.mapError((error) => targetError(target.id, "assess-model", error)))
    const result = assessment.results[0]
    const fit = result?._tag === "Assessed"
      ? result.profiles.find((profile) => profile._tag === "Fits")
      : undefined
    if (!fit || fit._tag !== "Fits") {
      return yield* new TargetError({
        targetId: target.id,
        operation: "assess-model",
        message: `ICN did not assess the requested serving profile as fitting: ${result?._tag ?? "missing result"}`,
      })
    }

    const response = yield* client.models.loadModelInstance({
      payload: { instanceId: load.instanceId, configuration: fit.configuration },
    }).pipe(Effect.mapError((error) => targetError(target.id, "load-model", error)))
    const events = yield* response.events.pipe(
      Stream.runCollect,
      Effect.mapError((error) => targetError(target.id, "load-model", error)),
    )
    const failed = Chunk.findFirst(events, (event) => event._tag === "Failed")
    if (Option.isSome(failed) && failed.value._tag === "Failed") {
      return yield* new TargetError({
        targetId: target.id,
        operation: "load-model",
        message: failed.value.failure.message,
      })
    }
    const ready = Chunk.findFirst(events, (event) => event._tag === "Ready")
    if (Option.isNone(ready) || ready.value._tag !== "Ready") {
      return yield* new TargetError({
        targetId: target.id,
        operation: "load-model",
        message: "ICN load stream completed without a Ready event",
      })
    }

    const allocation = ready.value.ready.allocation
    if (allocation.parallelSequences !== target.parallelSequences) {
      return yield* new TargetError({
        targetId: target.id,
        operation: "validate-capacity",
        message: `requested ${target.parallelSequences} parallel sequences but ICN allocated ${allocation.parallelSequences}`,
      })
    }

    yield* Effect.addFinalizer(() =>
      client.models.stopModelInstance({ path: { instance_id: load.instanceId } }).pipe(Effect.ignore),
    )
    return {
      servedModel: fit.configuration.id,
      instanceId: load.instanceId,
      parallelSequences: allocation.parallelSequences,
    }
  })

const acquireTarget = (
  target: TargetConfiguration,
): Effect.Effect<
  TargetSession,
  TargetError,
  CommandExecutor.CommandExecutor | EndpointClient | HttpClient.HttpClient | Scope.Scope
> =>
  Effect.gen(function* () {
    const baseEndpoint: EndpointConfiguration = {
      endpoint: target.endpoint,
      servedModel: target.servedModel,
      apiKey: target.apiKey,
      timeoutMs: 300_000,
      requestBody: target.requestBody,
    }
    if (target.kind === "existing") {
      yield* probeEndpoint(baseEndpoint).pipe(
        Effect.mapError((error) => targetError(target.id, "readiness", error)),
      )
      return {
        target,
        endpoint: baseEndpoint,
        rootPid: Option.none(),
        diagnostic: Effect.succeed(""),
      }
    }

    const output = yield* Ref.make("")
    const command = Command.make(target.executable, ...target.args).pipe(
      Command.env({ ...process.env, ...target.env }),
    )
    const processHandle = yield* Command.start(command).pipe(
      Effect.mapError((error) => targetError(target.id, "launch", error)),
    )
    yield* Effect.addFinalizer(() => stopProcess(target, processHandle))
    yield* processHandle.stdout.pipe(
      Stream.decodeText(),
      Stream.runForEach((chunk) => appendBounded(output, chunk)),
      Effect.ignore,
      Effect.forkScoped,
    )
    yield* processHandle.stderr.pipe(
      Stream.decodeText(),
      Stream.runForEach((chunk) => appendBounded(output, chunk)),
      Effect.ignore,
      Effect.forkScoped,
    )

    yield* waitForReady(target, processHandle, baseEndpoint).pipe(
      Effect.tapError(() => Ref.get(output).pipe(Effect.flatMap((diagnostic) =>
        Effect.logError(`Managed target ${target.id} failed readiness`).pipe(
          Effect.annotateLogs({ diagnostic: diagnostic.slice(-4_000) }),
        )))),
    )

    if (target.engine !== "icn") {
      return {
        target,
        endpoint: baseEndpoint,
        rootPid: Option.some(Number(processHandle.pid)),
        diagnostic: Ref.get(output),
      }
    }

    const provisioned = yield* provisionIcnModel(target)
    const sessionTarget: ManagedTarget = {
      ...target,
      servedModel: provisioned.servedModel,
      requestBody: { model_instance_id: provisioned.instanceId },
      parallelSequences: provisioned.parallelSequences,
    }
    return {
      target: sessionTarget,
      endpoint: {
        ...baseEndpoint,
        servedModel: provisioned.servedModel,
        requestBody: { model_instance_id: provisioned.instanceId },
      },
      rootPid: Option.some(Number(processHandle.pid)),
      diagnostic: Ref.get(output),
    }
  })

export interface TargetLauncherService {
  readonly open: (
    target: TargetConfiguration,
  ) => Effect.Effect<TargetSession, TargetError, Scope.Scope>
}

export class TargetLauncher extends Context.Tag("@magnitudedev/inference-benchmark/TargetLauncher")<
  TargetLauncher,
  TargetLauncherService
>() {}

export const TargetLauncherLive = Layer.effect(
  TargetLauncher,
  Effect.gen(function* () {
    const commandExecutor = yield* CommandExecutor.CommandExecutor
    const endpointClient = yield* EndpointClient
    const httpClient = yield* HttpClient.HttpClient
    return TargetLauncher.of({
      open: (target) => acquireTarget(target).pipe(
        Effect.provideService(CommandExecutor.CommandExecutor, commandExecutor),
        Effect.provideService(EndpointClient, endpointClient),
        Effect.provideService(HttpClient.HttpClient, httpClient),
      ),
    })
  }),
)

export const openTarget = (
  target: TargetConfiguration,
): Effect.Effect<TargetSession, TargetError, TargetLauncher | Scope.Scope> =>
  Effect.flatMap(TargetLauncher, (launcher) => launcher.open(target))

export interface ManagedComparisonOptions {
  readonly model: ModelIdentity
  readonly icnExecutable?: string
  readonly llamaExecutable?: string
  readonly port?: number
  readonly maxSequences?: number
}

const executable = (path: string | null | undefined) =>
  Effect.gen(function* () {
    if (!path) return Option.none<string>()
    const fs = yield* FileSystem.FileSystem
    const exists = yield* fs.exists(path).pipe(Effect.catchAll(() => Effect.succeed(false)))
    if (!exists) return Option.none<string>()
    const info = yield* fs.stat(path).pipe(Effect.option)
    return Option.isSome(info) && info.value.type === "File"
      ? Option.some(resolve(path))
      : Option.none<string>()
  })

const firstExecutable = (candidates: readonly (string | null | undefined)[]) =>
  Effect.gen(function* () {
    for (const candidate of candidates) {
      const found = yield* executable(candidate)
      if (Option.isSome(found)) return found
    }
    return Option.none<string>()
  })

export const resolveIcnExecutable = (
  explicit?: string,
): Effect.Effect<string, TargetError, FileSystem.FileSystem> =>
  firstExecutable([
    explicit,
    process.env.MAGNITUDE_ICN_SERVER,
    Bun.which("magnitude-icn"),
    "inference/target/benchmark-release/bin/magnitude-icn",
    "inference/target/development/bin/magnitude-icn",
  ]).pipe(Effect.flatMap(Option.match({
    onSome: Effect.succeed,
    onNone: () => Effect.fail(new TargetError({
      targetId: "icn",
      operation: "resolve-executable",
      message: "magnitude-icn was not found; build it or pass --icn-executable",
    })),
  })))

const managedLlamaMarker = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = resolve(homedir(), ".magnitude", "local-inference", "llamacpp", "distribution")
  const markerPath = join(root, "current.json")
  if (!(yield* fs.exists(markerPath).pipe(Effect.catchAll(() => Effect.succeed(false))))) {
    return Option.none<string>()
  }
  const marker = yield* fs.readFileString(markerPath).pipe(Effect.option)
  if (Option.isNone(marker)) return Option.none<string>()
  const decoded = yield* Effect.sync(() => {
    try {
      return JSON.parse(marker.value) as { readonly executables?: { readonly server?: unknown } }
    } catch {
      return undefined
    }
  })
  return typeof decoded?.executables?.server === "string"
    ? Option.some(resolve(root, decoded.executables.server))
    : Option.none<string>()
})

export const resolveLlamaCppExecutable = (
  explicit?: string,
): Effect.Effect<string, TargetError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const marker = yield* managedLlamaMarker
    const found = yield* firstExecutable([
      explicit,
      process.env.MAGNITUDE_LLAMA_SERVER,
      Option.getOrUndefined(marker),
      Bun.which("llama-server"),
      "inference/native/llama.cpp/build/bin/llama-server",
      "inference/native/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/build/bin/llama-server",
    ])
    return yield* Option.match(found, {
      onSome: Effect.succeed,
      onNone: () => Effect.fail(new TargetError({
        targetId: "llama.cpp",
        operation: "resolve-executable",
        message: "llama-server was not found; install it or pass --llama-executable",
      })),
    })
  })

export function managedIcnTarget(options: ManagedComparisonOptions): ManagedTarget {
  const port = options.port ?? 8091
  const executablePath = options.icnExecutable ?? resolve("inference/target/development/bin/magnitude-icn")
  return {
    kind: "managed",
    engine: "icn",
    id: "icn",
    executable: executablePath,
    endpoint: `http://127.0.0.1:${port}`,
    servedModel: options.model.id,
    readinessPath: "/health",
    args: [
      "serve",
      "--bind", `127.0.0.1:${port}`,
      "--installation", resolve(dirname(dirname(executablePath)), "installation.json"),
    ],
    modelLoad: {
      artifactSha256: options.model.artifactSha256,
      contextLimit: options.model.contextLimit,
      instanceId: "benchmark",
    },
    parallelSequences: options.maxSequences ?? 4,
  }
}

export function managedLlamaCppTarget(options: ManagedComparisonOptions): ManagedTarget {
  const port = options.port ?? 8091
  const parallelSequences = options.maxSequences ?? 4
  return {
    kind: "managed",
    engine: "llama.cpp",
    id: "llama.cpp",
    executable: options.llamaExecutable ?? "llama-server",
    endpoint: `http://127.0.0.1:${port}`,
    servedModel: options.model.id,
    readinessPath: "/health",
    args: [
      "--host", "127.0.0.1",
      "--port", String(port),
      "--model", options.model.artifactPath,
      "--alias", options.model.id,
      "--ctx-size", String(options.model.contextLimit * parallelSequences),
      "--parallel", String(parallelSequences),
      "--kv-unified",
      "--cont-batching",
    ],
    parallelSequences,
  }
}

export function existingTarget(
  id: string,
  endpoint: string,
  servedModel: string,
  apiKey?: string,
  parallelSequences = 1,
): ExistingTarget {
  return { kind: "existing", id, endpoint, servedModel, apiKey, parallelSequences }
}
