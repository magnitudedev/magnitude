import { createHash, randomUUID } from "node:crypto"
import * as FileSystem from "@effect/platform/FileSystem"
import * as HttpClient from "@effect/platform/HttpClient"
import * as Path from "@effect/platform/Path"
import { Context, Data, Effect, Either, Exit, Fiber, Option, Redacted, Ref, Schedule, Schema, Scope, Stream } from "effect"
import type { ModelFileId, ModelFileRegistryApi, ResolvedModelFiles } from "../model-files"
import { LlamaInstanceId, LlamaModelRegistrationId, type ExternalServerConfigId, type LlamaServedModelId } from "./identity"
import type { LlamaCli, LlamaRouterHost, RunningLlamaProcess } from "./cli"
import { renderExecutionProfilePreset, type LlamaExecutionProfile } from "./execution-profile"
import { LlamaServerFailureReason, makeLlamaServerClient, type LlamaInstanceObservation, type LlamaRouterController, type LlamaServerError, type LlamaServerObserver, type LlamaServedModelObservation } from "./server"

export interface LlamaInferenceTarget {
  readonly origin: URL
  readonly authorization: Option.Option<Redacted.Redacted<string>>
  readonly model: LlamaServedModelId
}

export interface LlamaModelLease {
  readonly instanceId: LlamaInstanceId
  readonly model: LlamaServedModelObservation
  readonly target: LlamaInferenceTarget
}

export interface ManagedModelRequest {
  readonly modelFileId: ModelFileId
  readonly servedModelId: LlamaServedModelId
  readonly profile: LlamaExecutionProfile
}

export interface ExternalModelRequest {
  readonly instanceId: LlamaInstanceId
  readonly servedModelId: LlamaServedModelId
}
export type LlamaModelRequest = Data.TaggedEnum<{
  Managed: { readonly request: ManagedModelRequest }
  External: { readonly request: ExternalModelRequest }
}>
export const LlamaModelRequest = Data.taggedEnum<LlamaModelRequest>()
export const LlamaAcquireFailureReason = Schema.Literal("instance-not-found", "not-already-served", "registration-conflict", "all-slots-leased", "catalog-change-in-use", "model-unavailable", "server-start-failed", "server-start-timeout", "process-exited", "load-rejected", "load-failed", "load-timeout")
export type LlamaAcquireFailureReason = Schema.Schema.Type<typeof LlamaAcquireFailureReason>
export const LlamaControlOperation = Schema.Literal("unload", "restart", "stop")
export type LlamaControlOperation = Schema.Schema.Type<typeof LlamaControlOperation>
export const LlamaControlFailureReason = Schema.Literal("model-not-registered", "server-rejected", "process-failure", "preset-write-failed")
export type LlamaControlFailureReason = Schema.Schema.Type<typeof LlamaControlFailureReason>
export const LlamaLeaseGuardOperation = Schema.Literal("unload", "restart", "stop", "evict")
export type LlamaLeaseGuardOperation = Schema.Schema.Type<typeof LlamaLeaseGuardOperation>
export interface LlamaInstanceEvent {
  readonly capturedAt: Date
  readonly observation: LlamaInstanceObservation
}
export class LlamaObservationError extends Data.TaggedError("LlamaObservationError")<{ readonly instanceId: LlamaInstanceId; readonly reason: Schema.Schema.Type<typeof LlamaServerFailureReason> }> {}
export class LlamaAcquireError extends Data.TaggedError("LlamaAcquireError")<{ readonly instanceId: LlamaInstanceId; readonly modelId: LlamaServedModelId; readonly reason: LlamaAcquireFailureReason }> {}
export class ModelNotAlreadyServed extends Data.TaggedError("ModelNotAlreadyServed")<{ readonly instanceId: LlamaInstanceId; readonly modelId: LlamaServedModelId }> {}
export class LlamaControlError extends Data.TaggedError("LlamaControlError")<{ readonly instanceId: LlamaInstanceId; readonly operation: LlamaControlOperation; readonly reason: LlamaControlFailureReason; readonly modelId: Option.Option<LlamaServedModelId> }> {}
export class ModelInUse extends Data.TaggedError("ModelInUse")<{ readonly instanceId: LlamaInstanceId; readonly operation: LlamaLeaseGuardOperation; readonly leases: number; readonly modelId: Option.Option<LlamaServedModelId> }> {}
export class LlamaInstanceNotFound extends Data.TaggedError("LlamaInstanceNotFound")<{ readonly id: LlamaInstanceId }> {}

export interface LlamaManagedInstance {
  readonly _tag: "Managed"
  readonly id: LlamaInstanceId
  readonly observe: Effect.Effect<LlamaInstanceObservation, LlamaObservationError>
  readonly acquire: (request: ManagedModelRequest) => Effect.Effect<LlamaModelLease, LlamaAcquireError, Scope.Scope>
  readonly unload: (model: LlamaServedModelId) => Effect.Effect<void, LlamaControlError | ModelInUse>
  readonly restart: Effect.Effect<void, LlamaControlError | ModelInUse>
  readonly stop: Effect.Effect<void, LlamaControlError | ModelInUse>
  readonly events: Stream.Stream<LlamaInstanceEvent, LlamaObservationError>
}

export interface LlamaExternalInstance {
  readonly _tag: "External"
  readonly id: LlamaInstanceId
  readonly observe: Effect.Effect<LlamaInstanceObservation, LlamaObservationError>
  readonly acquireExisting: (model: LlamaServedModelId) => Effect.Effect<LlamaModelLease, ModelNotAlreadyServed, Scope.Scope>
}
export type LlamaInstance = LlamaManagedInstance | LlamaExternalInstance
export interface LlamaInstanceSnapshot {
  readonly instances: readonly LlamaInstanceObservation[]
  readonly failures: readonly LlamaObservationError[]
  readonly capturedAt: Date
}

export interface LlamaInstanceRegistryApi {
  readonly inspect: Effect.Effect<LlamaInstanceSnapshot>
  readonly refreshExternal: Effect.Effect<LlamaInstanceSnapshot>
  readonly get: (id: LlamaInstanceId) => Effect.Effect<LlamaInstance, LlamaInstanceNotFound>
  readonly acquire: (request: LlamaModelRequest) => Effect.Effect<LlamaModelLease, LlamaAcquireError, Scope.Scope>
  readonly stopManaged: Effect.Effect<void, LlamaControlError | ModelInUse>
}
export class LlamaInstanceRegistry extends Context.Tag("@magnitudedev/local-inference/LlamaInstanceRegistry")<LlamaInstanceRegistry, LlamaInstanceRegistryApi>() {}
export interface ExternalLlamaServerConfig { readonly id: ExternalServerConfigId; readonly origin: URL; readonly authorization: Option.Option<Redacted.Redacted<string>>; readonly label: Option.Option<string> }
export interface LlamaInstanceRegistryOptions { readonly cli: LlamaCli; readonly modelFiles: ModelFileRegistryApi; readonly presetPath: string; readonly host: LlamaRouterHost; readonly port: number; readonly apiKey: Redacted.Redacted<string>; readonly modelsMax: number; readonly external: readonly ExternalLlamaServerConfig[] }

interface Registration { readonly id: LlamaModelRegistrationId; readonly request: ManagedModelRequest; readonly files: ResolvedModelFiles; readonly leases: number; readonly lastReleasedAt: number }
interface ManagedRuntime { readonly scope: Scope.CloseableScope; readonly process: RunningLlamaProcess; readonly observer: LlamaServerObserver; readonly controller: LlamaRouterController }
type ManagedRuntimeState = Data.TaggedEnum<{
  Stopped: Record<never, never>
  Starting: { readonly scope: Scope.CloseableScope }
  Running: { readonly runtime: ManagedRuntime }
  Failed: { readonly failure: LlamaAcquireError }
}>
const ManagedRuntimeState = Data.taggedEnum<ManagedRuntimeState>()
const registrationId = (request: ManagedModelRequest): LlamaModelRegistrationId => LlamaModelRegistrationId.make(createHash("sha256").update(`${request.modelFileId}\0${request.servedModelId}\0${request.profile.id}`).digest("hex"))
const observationError = (
  id: LlamaInstanceId,
  error: LlamaServerError,
): LlamaObservationError => {
  if (error.reason === "transport") {
    return new LlamaObservationError({ instanceId: id, reason: "transport" })
  }

  if (error.reason === "rejected") {
    return new LlamaObservationError({ instanceId: id, reason: "rejected" })
  }

  return new LlamaObservationError({ instanceId: id, reason: "invalid-response" })
}

const snapshotFromObservations = (
  results: readonly Either.Either<LlamaInstanceObservation, LlamaObservationError>[],
): LlamaInstanceSnapshot => {
  const instances: LlamaInstanceObservation[] = []
  const failures: LlamaObservationError[] = []

  for (const result of results) {
    if (Either.isRight(result)) instances.push(result.right)
    else failures.push(result.left)
  }

  return {
    instances,
    failures,
    capturedAt: new Date(),
  }
}

export const makeLlamaInstanceRegistry = (options: LlamaInstanceRegistryOptions): Effect.Effect<LlamaInstanceRegistryApi, never, FileSystem.FileSystem | Path.Path | HttpClient.HttpClient | Scope.Scope> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const http = yield* HttpClient.HttpClient
  const managedId = LlamaInstanceId.make("managed")
  const lock = yield* Effect.makeSemaphore(1)
  const registrations = yield* Ref.make<ReadonlyMap<LlamaServedModelId, Registration>>(new Map())
  const runtime = yield* Ref.make<ManagedRuntimeState>(ManagedRuntimeState.Stopped())
  const closeRuntimeState = ManagedRuntimeState.$match({
    Stopped: () => Effect.void,
    Starting: ({ scope }) => Scope.close(scope, Exit.void),
    Running: ({ runtime }) => Scope.close(runtime.scope, Exit.void),
    Failed: () => Effect.void,
  })
  yield* Effect.addFinalizer(() => Ref.get(runtime).pipe(Effect.flatMap(closeRuntimeState)))

  const makeClient = (origin: URL, authorization: Option.Option<Redacted.Redacted<string>>) => makeLlamaServerClient({ origin, authorization, timeout: Option.none() }).pipe(Effect.provideService(HttpClient.HttpClient, http))
  const makeExternal = (config: ExternalLlamaServerConfig): Effect.Effect<LlamaExternalInstance> => Effect.gen(function* () {
    const id = LlamaInstanceId.make(`external_${createHash("sha256").update(config.id).digest("hex")}`)
    const client = yield* makeClient(config.origin, config.authorization)
    const observe = client.observer.observe(id, "external").pipe(Effect.mapError((error) => observationError(id, error)))
    return { _tag: "External", id, observe, acquireExisting: (model) => Effect.acquireRelease(
      Effect.gen(function* () {
        const observation = yield* observe.pipe(Effect.mapError(() => new ModelNotAlreadyServed({ instanceId: id, modelId: model })))
        const found = Option.fromNullable(observation.models.find((candidate) => candidate.id === model && (candidate.status === "loaded" || candidate.status === "sleeping")))
        if (Option.isNone(found) || observation.health !== "ready") return yield* new ModelNotAlreadyServed({ instanceId: id, modelId: model })
        return { instanceId: id, model: found.value, target: { origin: config.origin, authorization: config.authorization, model } }
      }),
      () => Effect.void,
    ) }
  })
  const external = yield* Effect.forEach(options.external, makeExternal)

  const renderPreset = (entries: ReadonlyMap<LlamaServedModelId, Registration>): Effect.Effect<string, LlamaControlError> => Effect.gen(function* () {
    const sections = ["version = 1"]
    for (const registration of [...entries.values()].sort((left, right) => String(left.request.servedModelId).localeCompare(String(right.request.servedModelId)))) {
      const { request, files } = registration
      const modelPaths = [files.primaryPath, ...Option.toArray(files.projectorPath)]
      if (/[\[\]\r\n\0]/.test(request.servedModelId) || modelPaths.some((value) => /[\r\n\0]/.test(value))) return yield* new LlamaControlError({ instanceId: managedId, operation: "restart", reason: "preset-write-failed", modelId: Option.some(request.servedModelId) })
      const profile = request.profile
      const lines = [`[${request.servedModelId}]`, `model = ${files.primaryPath}`, "load-on-startup = false"]
      Option.map(files.projectorPath, (projectorPath) => lines.push(`mmproj = ${projectorPath}`))
      lines.push(...renderExecutionProfilePreset(profile))
      sections.push(lines.join("\n"))
    }
    return `${sections.join("\n\n")}\n`
  })
  const writePreset = (entries: ReadonlyMap<LlamaServedModelId, Registration>) => Effect.gen(function* () {
    const content = yield* renderPreset(entries)
    const directory = path.dirname(options.presetPath)
    const temporary = `${options.presetPath}.${randomUUID()}.tmp`
    yield* fs.makeDirectory(directory, { recursive: true }).pipe(Effect.mapError(() => new LlamaControlError({ instanceId: managedId, operation: "restart", reason: "preset-write-failed", modelId: Option.none() })))
    yield* Effect.acquireUseRelease(
      fs.writeFileString(temporary, content, { flag: "wx", mode: 0o600 }).pipe(Effect.mapError(() => new LlamaControlError({ instanceId: managedId, operation: "restart", reason: "preset-write-failed", modelId: Option.none() }))),
      () => fs.rename(temporary, options.presetPath).pipe(Effect.mapError(() => new LlamaControlError({ instanceId: managedId, operation: "restart", reason: "preset-write-failed", modelId: Option.none() }))),
      () => fs.remove(temporary, { force: true }).pipe(Effect.ignore),
    )
  })
  const stopRuntime = Effect.gen(function* () {
    const current = yield* Ref.getAndSet(runtime, ManagedRuntimeState.Stopped())
    yield* closeRuntimeState(current)
  })
  const startRuntime = (modelId: LlamaServedModelId): Effect.Effect<ManagedRuntime, LlamaAcquireError> => Effect.gen(function* () {
    const existing = yield* Ref.get(runtime)
    if (existing._tag === "Running") return existing.runtime
    const entries = yield* Ref.get(registrations)
    yield* writePreset(entries).pipe(Effect.mapError(() => new LlamaAcquireError({ instanceId: managedId, modelId, reason: "server-start-failed" })))
    const processScope = yield* Scope.make()
    yield* Ref.set(runtime, ManagedRuntimeState.Starting({ scope: processScope }))
    const boot = Effect.gen(function* () {
      const process = yield* options.cli.startRouter({ presetPath: options.presetPath, host: options.host, port: options.port, apiKey: options.apiKey, modelsMax: Option.some(options.modelsMax), modelSleepIdleSeconds: Option.none() }).pipe(Effect.provideService(Scope.Scope, processScope), Effect.mapError(() => new LlamaAcquireError({ instanceId: managedId, modelId, reason: "server-start-failed" })))
      const client = yield* makeClient(process.origin, Option.some(options.apiKey))
      const created = { scope: processScope, process, observer: client.observer, controller: client.controller }
      yield* Ref.set(runtime, ManagedRuntimeState.Running({ runtime: created }))
      const processFailure = new LlamaAcquireError({ instanceId: managedId, modelId, reason: "process-exited" })
      const exitFiber = yield* process.exited.pipe(
        Effect.exit,
        Effect.tap(() => Ref.update(runtime, (state) => state._tag === "Running" && state.runtime === created ? ManagedRuntimeState.Failed({ failure: processFailure }) : state)),
        Effect.forkDaemon,
      )
      yield* Effect.addFinalizer(() => Fiber.interruptFork(exitFiber)).pipe(Effect.provideService(Scope.Scope, processScope))
      for (let attempt = 0; attempt < 100; attempt++) {
        const exited = yield* Fiber.poll(exitFiber)
        if (Option.isSome(exited)) return yield* processFailure
        const health = yield* client.observer.health.pipe(Effect.either)
        if (health._tag === "Right" && health.right === "ready") return created
        yield* Effect.sleep("50 millis")
      }
      return yield* new LlamaAcquireError({ instanceId: managedId, modelId, reason: "server-start-timeout" })
    })
    return yield* boot.pipe(
      Effect.tapError((failure) => Scope.close(processScope, Exit.void).pipe(Effect.zipRight(Ref.set(runtime, ManagedRuntimeState.Failed({ failure }))))),
      Effect.onInterrupt(() => Scope.close(processScope, Exit.void).pipe(Effect.zipRight(Ref.set(runtime, ManagedRuntimeState.Stopped())))),
    )
  })
  const unavailableManagedObservation = (diagnostics: LlamaInstanceObservation["diagnostics"]): LlamaInstanceObservation => ({ id: managedId, ownership: "managed", health: "unavailable", mode: "unknown", build: Option.none(), capabilities: { models: "unknown", modelEvents: "unknown", load: "unknown", unload: "unknown", sleep: "unknown" }, models: [], diagnostics })
  const runningRuntime = (state: ManagedRuntimeState): Option.Option<ManagedRuntime> => state._tag === "Running" ? Option.some(state.runtime) : Option.none()
  const hasLiveRuntime = (state: ManagedRuntimeState): boolean => state._tag === "Starting" || state._tag === "Running"
  const observeManaged = Ref.get(runtime).pipe(Effect.flatMap(ManagedRuntimeState.$match({
    Stopped: () => Effect.succeed(unavailableManagedObservation([])),
    Starting: () => Effect.succeed(unavailableManagedObservation([])),
    Running: ({ runtime }) => runtime.observer.observe(managedId, "managed").pipe(Effect.mapError((error) => observationError(managedId, error))),
    Failed: ({ failure }) => Effect.succeed(unavailableManagedObservation([{ code: "managed_runtime_failed", message: failure.reason, modelId: Option.some(failure.modelId) }])),
  })))
  const controlError = (operation: LlamaControlError["operation"], modelId: Option.Option<LlamaServedModelId>) => new LlamaControlError({ instanceId: managedId, operation, reason: "server-rejected", modelId })
  const replaceLeaseCount = (registration: Registration, leases: number, lastReleasedAt: number): Registration => ({
    id: registration.id,
    request: registration.request,
    files: registration.files,
    leases,
    lastReleasedAt,
  })
  const acquireLease = (modelId: LlamaServedModelId) => Ref.update(registrations, (entries) => Option.match(
    Option.fromNullable(entries.get(modelId)),
    {
      onNone: () => entries,
      onSome: (registration) => new Map(entries).set(modelId, replaceLeaseCount(registration, registration.leases + 1, registration.lastReleasedAt)),
    },
  ))
  const releaseLease = (modelId: LlamaServedModelId) => Ref.update(registrations, (entries) => Option.match(
    Option.fromNullable(entries.get(modelId)),
    {
      onNone: () => entries,
      onSome: (registration) => new Map(entries).set(modelId, replaceLeaseCount(registration, Math.max(0, registration.leases - 1), Date.now())),
    },
  ))
  const isLoaded = (model: LlamaServedModelObservation): boolean => model.status === "loaded" || model.status === "sleeping"
  const managed: LlamaManagedInstance = {
    _tag: "Managed", id: managedId, observe: observeManaged,
    events: Stream.repeatEffectWithSchedule(observeManaged, Schedule.spaced("1 second")).pipe(Stream.map((observation) => ({ capturedAt: new Date(), observation }))),
    acquire: (request) => Effect.acquireRelease(
      lock.withPermits(1)(Effect.gen(function* () {
        let entries = yield* Ref.get(registrations)
        const existingRegistration = Option.fromNullable(entries.get(request.servedModelId))
        if (Option.exists(existingRegistration, (registration) => registration.request.modelFileId !== request.modelFileId || registration.request.profile.id !== request.profile.id)) return yield* new LlamaAcquireError({ instanceId: managedId, modelId: request.servedModelId, reason: "registration-conflict" })
        if (Option.isNone(existingRegistration)) {
          if (entries.size >= options.modelsMax) {
            const evictable = Option.fromNullable([...entries.values()].filter(({ leases }) => leases === 0).sort((left, right) => left.lastReleasedAt - right.lastReleasedAt || String(left.request.servedModelId).localeCompare(String(right.request.servedModelId)))[0])
            if (Option.isNone(evictable)) return yield* new LlamaAcquireError({ instanceId: managedId, modelId: request.servedModelId, reason: "all-slots-leased" })
            const current = yield* Ref.get(runtime)
            const running = runningRuntime(current)
            if (Option.isSome(running)) yield* running.value.controller.unload(evictable.value.request.servedModelId).pipe(Effect.mapError(() => new LlamaAcquireError({ instanceId: managedId, modelId: request.servedModelId, reason: "load-rejected" })))
            const withoutEvicted = new Map(entries)
            withoutEvicted.delete(evictable.value.request.servedModelId)
            entries = withoutEvicted
          }
          if (hasLiveRuntime(yield* Ref.get(runtime)) && [...entries.values()].some(({ leases }) => leases > 0)) return yield* new LlamaAcquireError({ instanceId: managedId, modelId: request.servedModelId, reason: "catalog-change-in-use" })
          const files = yield* options.modelFiles.resolve(request.modelFileId).pipe(Effect.mapError(() => new LlamaAcquireError({ instanceId: managedId, modelId: request.servedModelId, reason: "model-unavailable" })))
          const registration = { id: registrationId(request), request, files, leases: 0, lastReleasedAt: 0 }
          entries = new Map(entries).set(request.servedModelId, registration)
          yield* Ref.set(registrations, entries)
          yield* stopRuntime
        } else yield* options.modelFiles.resolve(request.modelFileId).pipe(Effect.mapError(() => new LlamaAcquireError({ instanceId: managedId, modelId: request.servedModelId, reason: "model-unavailable" })))
        const current = yield* startRuntime(request.servedModelId)
        const before = yield* current.observer.models.pipe(Effect.mapError(() => new LlamaAcquireError({ instanceId: managedId, modelId: request.servedModelId, reason: "load-rejected" })))
        const loaded = Option.fromNullable(before.find(({ id }) => id === request.servedModelId))
        if (!Option.exists(loaded, isLoaded)) yield* current.controller.load(request.servedModelId).pipe(Effect.mapError(() => new LlamaAcquireError({ instanceId: managedId, modelId: request.servedModelId, reason: "load-rejected" })))
        let observed = loaded
        for (let attempt = 0; attempt < 300 && !Option.exists(observed, isLoaded); attempt++) {
          yield* Effect.sleep("100 millis")
          const models = yield* current.observer.models.pipe(Effect.mapError(() => new LlamaAcquireError({ instanceId: managedId, modelId: request.servedModelId, reason: "load-rejected" })))
          observed = Option.fromNullable(models.find(({ id }) => id === request.servedModelId))
          if (Option.exists(observed, ({ status }) => status === "failed")) return yield* new LlamaAcquireError({ instanceId: managedId, modelId: request.servedModelId, reason: "load-failed" })
        }
        const readyModel = yield* Option.match(observed, {
          onNone: () => Effect.fail(new LlamaAcquireError({ instanceId: managedId, modelId: request.servedModelId, reason: "load-timeout" })),
          onSome: (model) => isLoaded(model)
            ? Effect.succeed(model)
            : Effect.fail(new LlamaAcquireError({ instanceId: managedId, modelId: request.servedModelId, reason: "load-timeout" })),
        })
        yield* acquireLease(request.servedModelId)
        return { instanceId: managedId, model: readyModel, target: { origin: current.process.origin, authorization: Option.some(options.apiKey), model: request.servedModelId } }
      })),
      () => lock.withPermits(1)(releaseLease(request.servedModelId)),
    ),
    unload: (model) => lock.withPermits(1)(Effect.gen(function* () {
      const entries = yield* Ref.get(registrations)
      const registration = Option.fromNullable(entries.get(model))
      if (Option.isNone(registration)) return yield* new LlamaControlError({ instanceId: managedId, operation: "unload", reason: "model-not-registered", modelId: Option.some(model) })
      if (registration.value.leases > 0) return yield* new ModelInUse({ instanceId: managedId, operation: "unload", leases: registration.value.leases, modelId: Option.some(model) })
      const current = yield* Ref.get(runtime)
      const running = runningRuntime(current)
      if (Option.isSome(running)) yield* running.value.controller.unload(model).pipe(Effect.mapError(() => controlError("unload", Option.some(model))))
      const updated = new Map(entries)
      updated.delete(model)
      yield* Ref.set(registrations, updated)
      yield* writePreset(updated)
    })),
    restart: lock.withPermits(1)(Effect.gen(function* () {
      const entries = yield* Ref.get(registrations)
      const leases = [...entries.values()].reduce((sum, item) => sum + item.leases, 0)
      if (leases > 0) return yield* new ModelInUse({ instanceId: managedId, operation: "restart", leases, modelId: Option.none() })
      yield* stopRuntime
      const first = Option.fromNullable(entries.values().next().value)
      if (Option.isSome(first)) yield* startRuntime(first.value.request.servedModelId).pipe(Effect.mapError(() => new LlamaControlError({ instanceId: managedId, operation: "restart", reason: "process-failure", modelId: Option.none() })))
    })),
    stop: lock.withPermits(1)(Effect.gen(function* () {
      const entries = yield* Ref.get(registrations)
      const leases = [...entries.values()].reduce((sum, item) => sum + item.leases, 0)
      if (leases > 0) return yield* new ModelInUse({ instanceId: managedId, operation: "stop", leases, modelId: Option.none() })
      yield* stopRuntime
    })),
  }
  const instances = new Map<LlamaInstanceId, LlamaInstance>([[managedId, managed], ...external.map((instance) => [instance.id, instance] as const)])
  const observeInstances = (
    selected: Iterable<LlamaInstance>,
  ): Effect.Effect<LlamaInstanceSnapshot> => Effect.forEach(
    selected,
    (instance) => instance.observe.pipe(Effect.either),
    { concurrency: 4 },
  ).pipe(Effect.map(snapshotFromObservations))

  const inspect = observeInstances(instances.values())
  const inspectExternal = observeInstances(external)
  return {
    inspect,
    refreshExternal: inspectExternal,
    get: (id) => Option.match(Option.fromNullable(instances.get(id)), {
      onNone: () => Effect.fail(new LlamaInstanceNotFound({ id })),
      onSome: Effect.succeed,
    }),
    acquire: LlamaModelRequest.$match({
      Managed: ({ request }) => managed.acquire(request),
      External: ({ request }) => Effect.gen(function* () {
        const instance = Option.fromNullable(instances.get(request.instanceId))
        if (Option.isNone(instance) || instance.value._tag !== "External") return yield* new LlamaAcquireError({ instanceId: request.instanceId, modelId: request.servedModelId, reason: "instance-not-found" })
        return yield* instance.value.acquireExisting(request.servedModelId).pipe(Effect.mapError(() => new LlamaAcquireError({ instanceId: request.instanceId, modelId: request.servedModelId, reason: "not-already-served" })))
      }),
    }),
    stopManaged: managed.stop,
  }
})
