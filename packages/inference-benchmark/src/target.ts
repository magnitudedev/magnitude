import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import { makeIcnApiClient } from "@magnitudedev/icn-protocol/client"
import { Context, Data, Effect, Layer, Option, Ref, Schedule, Schema, Scope, Stream } from "effect"
import { homedir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { createWriteStream } from "node:fs"
import type {
  ExistingTarget,
  ManagedTarget,
  LogicalModelIdentity,
  TargetConfiguration,
} from "./domain"
import {
  EndpointClient,
  type EndpointConfiguration,
  probeEndpoint,
} from "./transport"
import { processTreePids } from "./memory"

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

const OmlxReadinessSchema = Schema.Struct({
  ready: Schema.Literal(true),
  discovered_model: Schema.String,
  served_model: Schema.String,
  loaded: Schema.Literal(true),
  context_capacity: Schema.Int.pipe(Schema.greaterThan(0)),
  max_concurrent_requests: Schema.Int.pipe(Schema.greaterThan(0)),
  speculative_backend: Schema.Literal("none", "mtp", "dflash", "dspark"),
  qualification_completed: Schema.Literal(true),
})

const probeOmlxReadiness = (
  target: ManagedTarget,
  endpoint: EndpointConfiguration,
): Effect.Effect<void, TargetError> => Effect.tryPromise({
  try: async () => {
    if (target.readiness.kind !== "omlx") throw new Error("invalid oMLX readiness policy")
    const response = await fetch(`${endpoint.endpoint.replace(/\/+$/, "")}/${target.readiness.path.replace(/^\/+/, "")}`)
    if (!response.ok) throw new Error(`readiness returned ${response.status}`)
    const status = Schema.decodeUnknownSync(OmlxReadinessSchema)(await response.json())
    const expected = target.readiness.expected
    if (status.discovered_model !== expected.servedModel || status.served_model !== expected.servedModel) {
      throw new Error(`expected model ${expected.servedModel}, received ${status.discovered_model}/${status.served_model}`)
    }
    if (status.context_capacity !== expected.contextCapacity) {
      throw new Error(`expected context ${expected.contextCapacity}, received ${status.context_capacity}`)
    }
    if (status.max_concurrent_requests !== expected.parallelSequences) {
      throw new Error(`expected concurrency ${expected.parallelSequences}, received ${status.max_concurrent_requests}`)
    }
    if (status.speculative_backend !== expected.speculativeBackend) {
      throw new Error(`expected ${expected.speculativeBackend}, received ${status.speculative_backend}`)
    }
  },
  catch: (error) => targetError(target.id, "readiness-pending", error),
})

const stopProcess = (
  target: ManagedTarget,
  processHandle: CommandExecutor.Process,
): Effect.Effect<void> =>
  Effect.gen(function* () {
    if (!(yield* processHandle.isRunning)) return
    const rootPid = Number(processHandle.pid)
    const initialTree = [...new Set([rootPid, ...processTreePids(rootPid)])]
    yield* processHandle.kill("SIGINT").pipe(Effect.ignore)
    const stopped = yield* Effect.sync(() => initialTree.every((pid) => {
      try {
        process.kill(pid, 0)
        return false
      } catch {
        return true
      }
    })).pipe(
      Effect.flatMap((done) => done ? Effect.succeed(true) : Effect.fail(undefined)),
      Effect.retry(Schedule.spaced("100 millis").pipe(Schedule.compose(Schedule.recurs(99)))),
      Effect.catchAll(() => Effect.succeed(false)),
    )
    if (stopped) return
    yield* Effect.sync(() => {
      const remaining = new Set([...initialTree, ...processTreePids(rootPid)])
      for (const pid of [...remaining].sort((left, right) => right - left)) {
        if (pid <= 1) continue
        try { process.kill(pid, "SIGKILL") } catch { /* already stopped */ }
      }
    })
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
    if (target.readiness.kind === "omlx") yield* probeOmlxReadiness(target, endpoint)
    else {
      yield* probeEndpoint(endpoint, target.readiness.path).pipe(
        Effect.mapError((error) => targetError(target.id, "readiness-pending", error)),
      )
    }
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
    const instance = yield* client.models.ensureModelInstance({
      payload: { modelId: target.servedModel },
    }).pipe(Effect.mapError((error) => targetError(target.id, "load-model", error)))
    if (instance.lifecycle._tag !== "Ready") {
      return yield* new TargetError({
        targetId: target.id,
        operation: "load-model",
        message: `ICN returned a non-ready instance: ${instance.lifecycle._tag}`,
      })
    }
    const allocation = instance.lifecycle.allocation
    if (allocation.parallelSequences !== target.parallelSequences) {
      return yield* new TargetError({
        targetId: target.id,
        operation: "validate-capacity",
        message: `requested ${target.parallelSequences} parallel sequences but ICN allocated ${allocation.parallelSequences}`,
      })
    }

    yield* Effect.addFinalizer(() =>
      client.models.stopModelInstance({ path: { instance_id: instance.id } }).pipe(Effect.ignore),
    )
    return {
      servedModel: target.servedModel,
      instanceId: instance.id,
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
      timeoutMs: target.requestTimeoutMs ?? 300_000,
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
      target.cwd ? Command.workingDirectory(target.cwd) : (value) => value,
    )
    const processHandle = yield* Command.start(command).pipe(
      Effect.mapError((error) => targetError(target.id, "launch", error)),
    )
    const log = target.logPath ? createWriteStream(target.logPath, { flags: "a" }) : undefined
    if (log) yield* Effect.addFinalizer(() => Effect.async<void>((resume) => {
      log.end(() => resume(Effect.void))
    }))
    yield* Effect.addFinalizer(() => stopProcess(target, processHandle))
    yield* processHandle.stdout.pipe(
      Stream.decodeText(),
      Stream.runForEach((chunk) => appendBounded(output, chunk).pipe(
        Effect.tap(() => Effect.sync(() => { log?.write(chunk) })),
      )),
      Effect.ignore,
      Effect.forkScoped,
    )
    yield* processHandle.stderr.pipe(
      Stream.decodeText(),
      Stream.runForEach((chunk) => appendBounded(output, chunk).pipe(
        Effect.tap(() => Effect.sync(() => { log?.write(chunk) })),
      )),
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
      parallelSequences: provisioned.parallelSequences,
    }
    return {
      target: sessionTarget,
      endpoint: {
        ...baseEndpoint,
        servedModel: provisioned.servedModel,
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
  readonly model: LogicalModelIdentity & { readonly artifactPath: string; readonly artifactSha256: string }
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
    readiness: { kind: "http", path: "/health" },
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
    readiness: { kind: "http", path: "/health" },
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
