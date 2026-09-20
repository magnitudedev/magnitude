import { FetchHttpClient, FileSystem } from "@effect/platform"
import { BunContext, BunHttpServer, BunRuntime } from "@effect/platform-bun"
import { Config, Console, Context, Effect, Layer, Option, Redacted, Schema } from "effect"
import { Authenticator, bearerAuthenticator } from "./api"
import { EntraAuthConfig, entraAuthenticator } from "./entra-auth"
import { GitHubAuthConfig, githubAuthenticator } from "./github-auth"
import { fileArtifactStore } from "./artifact-store"
import { startCoordinator, CoordinatorConfig } from "./coordinator"
import { databaseLayer } from "./database"
import { InfrastructureFailure, InvalidInput, Principal, Provider } from "./domain"
import { MachineAllocator, WorkerTransport } from "./machines"
import { ProcessExecutorLive } from "./process"
import { azureArtifactStore, AzureArtifactConfig } from "./providers/azure-artifacts"
import { namespaceAllocator, NamespaceImage, namespaceTransport } from "./providers/namespace"
import { assertRuntime } from "./runtime"
import { MachineProviders, WorkerRunner } from "./scheduler"
import { GuestRuntime, transportWorkerRunner, WorkerTransports } from "./worker-runner"
import { InputRegistryLive } from "./inputs"
import { AzureConfig, azureAllocator } from "./providers/azure"
import { azureLinuxBootstrap } from "./providers/azure-bootstrap"
import { outwardWorkerRunner, WorkerBootstraps } from "./outward-runner"
import { WorkerTicketsLive } from "./worker-tickets"
import { WorkerResultsLive } from "./worker-results"

export const ServerConfig = Schema.Struct({ coordinator: CoordinatorConfig,
  hostname: Schema.Literal("127.0.0.1", "::1", "0.0.0.0"), port: Schema.Int.pipe(Schema.between(0, 65535)),
  credentials: Schema.Array(Schema.Struct({ tokenEnvironment: Schema.String.pipe(Schema.pattern(/^[A-Z][A-Z0-9_]*$/)), principal: Principal })),
  storage: Schema.Union(Schema.Struct({ kind: Schema.Literal("file"), directory: Schema.NonEmptyString }), Schema.Struct({ kind: Schema.Literal("azure"), config: AzureArtifactConfig })),
  namespace: Schema.optionalWith(Schema.Struct({ executable: Schema.NonEmptyString, images: Schema.NonEmptyArray(NamespaceImage) }), { as: "Option", exact: true }),
  azure: Schema.optionalWith(Schema.Struct({ allocation: AzureConfig, workerOrigin: Schema.NonEmptyString }), { as: "Option", exact: true }),
  entra: Schema.optionalWith(EntraAuthConfig, { as: "Option", exact: true }),
  github: Schema.optionalWith(GitHubAuthConfig, { as: "Option", exact: true }),
  runtimes: Schema.Array(GuestRuntime),
})
export const configuredCoordinator = Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const config = yield* fs.readFileString(yield* Config.string("LAB_COORDINATOR_CONFIG")).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(ServerConfig))))
  if (config.runtimes.some(runtime => runtime.provider === "namespace" ? Option.isNone(config.namespace)
    : runtime.provider === "azure" ? Option.isNone(config.azure) || !runtime.artifactHost.startsWith("linux-") : true)) return yield* new InvalidInput({ message: "A runtime requires its configured provider; this entry point supports Namespace and Azure Linux execution" })
  for (const provider of ["namespace", "azure"] as const) if (Option.isSome<unknown>(config[provider]) && !config.runtimes.some(runtime => runtime.provider === provider)) return yield* new InvalidInput({ message: `${provider} allocation requires a configured guest runtime` })
  if (new Set(config.runtimes.map(runtime => `${runtime.provider}/${runtime.artifactHost}`)).size !== config.runtimes.length) return yield* new InvalidInput({ message: "Guest runtime identities must be unique" })
  if (config.credentials.length === 0 && Option.isNone(config.github) && Option.isNone(config.entra)) return yield* new InvalidInput({ message: "Configure at least one authentication method" })
  const credentials = yield* Effect.forEach(config.credentials, entry => Effect.gen(function* () {
    const token = yield* Config.redacted(entry.tokenEnvironment)
    if (Redacted.value(token).length < 32) return yield* new InvalidInput({ message: "Coordinator bearer credentials must contain at least 32 characters" })
    return { token, principal: entry.principal }
  }))
  if (new Set(credentials.map(credential => Redacted.value(credential.token))).size !== credentials.length) return yield* new InvalidInput({ message: "Each configured identity requires a distinct credential" })
  const bearer = Context.get(yield* Layer.build(bearerAuthenticator(credentials)), Authenticator)
  const github = Option.isSome(config.github) ? Option.some(Context.get(yield* Layer.build(githubAuthenticator(config.github.value)), Authenticator)) : Option.none()
  const entra = Option.isSome(config.entra) ? Option.some(Context.get(yield* Layer.build(entraAuthenticator(config.entra.value)), Authenticator)) : Option.none()
  const authentication = Layer.succeed(Authenticator, { authenticate: header => bearer.authenticate(header).pipe(
    Effect.catchAll(error => Option.isSome(github) ? github.value.authenticate(header) : Effect.fail(error)),
    Effect.catchAll(error => Option.isSome(entra) ? entra.value.authenticate(header) : Effect.fail(error)),
  ) })
  const database = databaseLayer(yield* Config.redacted("LAB_DATABASE_URL"))
  const storage = config.storage.kind === "file" ? fileArtifactStore(config.storage.directory) : azureArtifactStore(config.storage.config).pipe(Layer.provide(FetchHttpClient.layer))
  const allocators = new Map<typeof Provider.Type, MachineAllocator>(), transports = new Map<typeof Provider.Type, WorkerTransport>()
  if (Option.isSome(config.namespace)) {
    const namespace = config.namespace.value
    allocators.set("namespace", Context.get(yield* Layer.build(namespaceAllocator(namespace.executable, namespace.images)), MachineAllocator))
    transports.set("namespace", Context.get(yield* Layer.build(namespaceTransport(namespace.executable)), WorkerTransport))
  }
  if (Option.isSome(config.azure)) allocators.set("azure", Context.get(yield* Layer.build(azureAllocator(config.azure.value.allocation)), MachineAllocator))
  const runner = Layer.scoped(WorkerRunner, Effect.gen(function* () {
    const byProvider = new Map<typeof Provider.Type, WorkerRunner>()
    if (Option.isSome(config.namespace)) byProvider.set("namespace", Context.get(yield* Layer.build(transportWorkerRunner(config.runtimes.filter(runtime => runtime.provider === "namespace")).pipe(
      Layer.provide(Layer.succeed(WorkerTransports, { transports })))), WorkerRunner))
    if (Option.isSome(config.azure)) {
      const bootstrap = yield* azureLinuxBootstrap(config.azure.value.allocation)
      byProvider.set("azure", Context.get(yield* Layer.build(outwardWorkerRunner({ origin: config.azure.value.workerOrigin,
        runtimes: config.runtimes.filter(runtime => runtime.provider === "azure"), pollMs: 1000 }).pipe(
        Layer.provide(Layer.succeed(WorkerBootstraps, { providers: new Map([["azure" as const, bootstrap]]) })))), WorkerRunner))
    }
    return { run: (machine, assignment) => {
      const selected = byProvider.get(machine.provider)
      return selected ? selected.run(machine, assignment) : Effect.fail(new InfrastructureFailure({ operation: "worker-runner", message: "No configured runner for allocated provider" }))
    } } satisfies WorkerRunner
  }))
  const inputs = InputRegistryLive.pipe(Layer.provide(Layer.merge(database, storage)))
  const tickets = WorkerTicketsLive.pipe(Layer.provide(database))
  const results = WorkerResultsLive.pipe(Layer.provide(Layer.merge(database, tickets)))
  const services = Layer.mergeAll(database, storage, authentication, Layer.succeed(MachineProviders, { allocators }),
    runner.pipe(Layer.provide(Layer.mergeAll(inputs, tickets, results)))).pipe(Layer.provideMerge(BunHttpServer.layer({ hostname: config.hostname, port: config.port, maxRequestBodySize: 4 * 1024 ** 3, idleTimeout: 255 })))
  // The runner and HTTP input registry use the same durable database and object store.
  return yield* startCoordinator(config.coordinator).pipe(Effect.provide(yield* Layer.build(services)))
})
export const serveConfiguredCoordinator = Effect.scoped(Effect.gen(function* () {
  const coordinator = yield* configuredCoordinator
  yield* Console.log(`Magnitude testing coordinator listening on ${coordinator.address._tag === "TcpAddress" ? `${coordinator.address.hostname}:${coordinator.address.port}` : coordinator.address.path}`)
  yield* coordinator.run
}))
if (import.meta.main) BunRuntime.runMain(serveConfiguredCoordinator.pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
