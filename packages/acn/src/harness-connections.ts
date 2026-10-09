import { CommandExecutor, FileSystem, HttpClient, Path } from "@effect/platform"
import { Context, Effect, Layer, Option, PubSub, Schedule, Stream } from "effect"
import { homedir } from "node:os"
import { dirname, join, resolve } from "node:path"
import {
  HarnessConnectionFailed,
  type HarnessConnectOutcome,
  type HarnessConnectRequest,
  type HarnessConnectionsSnapshot,
  type HarnessId,
  type HarnessSetup,
  type HarnessSetupRequest,
} from "@magnitudedev/acn-protocol"
import {
  harnessCommandExecutor,
  harnessExecutableSearchPath,
  makeHarnessConnectionService,
  resolveHarnessConnectionPaths,
  resolveHarnessEnvironment,
  HarnessConnectionError,
} from "@magnitudedev/harness-connections"
import { BunSqliteDriverLayer } from "@magnitudedev/storage/sqlite/bun"
import { guardedCommandLayer } from "@magnitudedev/utils/guarded-command"
import { AcnHost } from "./server-settings"

export interface AcnHarnessConnectionsApi {
  /** Inspects on subscribe, after any connect or disconnect from any client, and every two seconds while observed. */
  readonly watch: Stream.Stream<HarnessConnectionsSnapshot>
  readonly connect: (request: HarnessConnectRequest) => Effect.Effect<HarnessConnectOutcome, HarnessConnectionFailed>
  readonly sync: (harness: Option.Option<HarnessId>) => Effect.Effect<void, HarnessConnectionFailed>
  readonly disconnect: (harness: HarnessId) => Effect.Effect<void, HarnessConnectionFailed>
  /** Another device sends the key this service enforces; this computer needs none. */
  readonly describe: (request: HarnessSetupRequest) => Effect.Effect<HarnessSetup, HarnessConnectionFailed>
}
export class AcnHarnessConnections extends Context.Tag("AcnHarnessConnections")<AcnHarnessConnections, AcnHarnessConnectionsApi>() {}

const failed = (error: HarnessConnectionError) => new HarnessConnectionFailed({
  operation: error.operation,
  harness: Option.fromNullable(error.harness),
  message: error.message,
})

/** The installed layout keeps the protected command helper beside the native host. */
const commandHelperPath = () => join(dirname(process.env.MAGNITUDE_NATIVE_HOST ?? join(dirname(process.execPath), "desktop-host.node")), "magnitude-command")

export const AcnHarnessConnectionsLive = Layer.scoped(AcnHarnessConnections, Effect.gen(function* () {
  const host = yield* AcnHost
  const context = yield* Effect.context<FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor | HttpClient.HttpClient>()
  const changes = yield* PubSub.unbounded<void>()
  // Harnesses are found through the user's login-shell environment, which a service started by
  // the desktop or a service manager does not inherit.
  const environment = yield* resolveHarnessEnvironment().pipe(
    Effect.provide(guardedCommandLayer(commandHelperPath())),
    Effect.orElseSucceed(() => Object.fromEntries(Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined))),
    Effect.cached,
  )
  const isolated = resolve(host.dataDir) !== resolve(join(homedir(), ".magnitude"))
  const service = yield* Effect.cached(Effect.gen(function* () {
    const values = yield* environment
    const executor = yield* harnessCommandExecutor(values)
    return yield* makeHarnessConnectionService({
      paths: yield* resolveHarnessConnectionPaths(isolated ? join(host.dataDir, "harness-home") : undefined, values),
      serviceEndpoint: `http://127.0.0.1:${host.port}`,
      detect: connector => connector.detect(harnessExecutableSearchPath(values.PATH)),
    }).pipe(Effect.provideService(CommandExecutor.CommandExecutor, executor))
  }).pipe(Effect.provide(Layer.mergeAll(Layer.succeedContext(context), BunSqliteDriverLayer))))
  const changed = PubSub.publish(changes, undefined).pipe(Effect.asVoid)
  // Running as the Linux server's service account, configuration written here would land in its
  // home, where no one's harnesses read it. People connect harnesses from their own terminal.
  const serverService = process.env.MAGNITUDE_SERVER_SERVICE === "1"
  const refuseOnServer = (operation: "connect" | "sync" | "disconnect", harness: Option.Option<HarnessId>) => serverService
    ? Effect.fail(new HarnessConnectionError({ operation, harness: Option.getOrUndefined(harness), message:
      "This Magnitude runs as a server. Connect a harness on this machine from your own terminal with `magnitude connections connect <harness>`, or copy its setup to another computer." }))
    : Effect.void

  const inspect = service.pipe(
    Effect.flatMap(connections => connections.inspect),
    Effect.map((connections): HarnessConnectionsSnapshot => ({ _tag: "Ready", connections })),
    Effect.catchAll(error => Effect.succeed<HarnessConnectionsSnapshot>({ _tag: "Unavailable", message: error.message })),
  )
  return AcnHarnessConnections.of({
    watch: Stream.concat(Stream.succeed(undefined), Stream.merge(Stream.fromPubSub(changes), Stream.fromSchedule(Schedule.spaced("2 seconds")))).pipe(
      Stream.mapEffect(() => inspect),
    ),
    connect: request => refuseOnServer("connect", Option.some(request.harness)).pipe(Effect.zipRight(service),
      Effect.flatMap(connections => connections.connect(request.harness, { model: request.model, installSkill: request.installSkill, launchOnStartup: false })),
      Effect.mapError(failed),
      Effect.map((result): HarnessConnectOutcome => ({
        companion: Option.map(result.companion, companion => ({
          name: companion.name,
          source: companion.source,
          securityNotice: companion.securityNotice,
          status: companion.status,
          activationInstructions: companion.activationInstructions,
        })),
        skillInstalled: result.skillInstalled,
      })),
      Effect.ensuring(changed),
    ),
    sync: harness => refuseOnServer("sync", harness).pipe(Effect.zipRight(service),
      Effect.flatMap(connections => connections.sync(Option.getOrUndefined(harness))),
      Effect.mapError(failed),
      Effect.asVoid,
      Effect.ensuring(changed),
    ),
    disconnect: harness => refuseOnServer("disconnect", Option.some(harness)).pipe(Effect.zipRight(service),
      Effect.flatMap(connections => connections.disconnect(harness)),
      Effect.mapError(failed),
      Effect.ensuring(changed),
    ),
    describe: request => {
      const key = request.remote ? host.activeNetwork.apiKey : Option.none<string>()
      return service.pipe(
        Effect.flatMap(connections => connections.describe({ harness: request.harness, model: request.model, platform: request.platform, origin: request.origin, key })),
        Effect.mapError(failed),
      )
    },
  })
}))
