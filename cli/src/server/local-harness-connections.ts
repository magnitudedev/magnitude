import { CommandExecutor, FetchHttpClient, HttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { join } from "node:path"
import { harnessCommandExecutor, makeHarnessConnectionService, resolveHarnessConnectionPaths } from "@magnitudedev/harness-connections"
import { BunSqliteDriverLayer } from "@magnitudedev/storage/sqlite/bun"
import { desktopDataDirectory, desktopIsolatedProfile, desktopServiceOrigin } from "./application"
import { NoServiceRunning } from "./acn-connection"

/**
 * Connecting needs the models the service on this machine offers. It is reached on loopback, not
 * through the owner socket, which a new member of the server's group cannot use until they log in again.
 */
export const requireLocalService = HttpClient.get(`${desktopServiceOrigin}/health`).pipe(
  Effect.timeout("3 seconds"), Effect.asVoid, Effect.mapError(() => new NoServiceRunning()), Effect.provide(FetchHttpClient.layer))

/**
 * Harness connections for the person running the CLI, written by this process into their own home
 * and pointing at the service on this machine. The service may run as another account (the Linux
 * server's `magnitude` user), so it never writes them for the CLI.
 */
export const localHarnessConnections = Effect.gen(function* () {
  const environment = Object.fromEntries(Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined))
  const executor = yield* harnessCommandExecutor(environment)
  return yield* makeHarnessConnectionService({
    paths: yield* resolveHarnessConnectionPaths(desktopIsolatedProfile ? join(desktopDataDirectory, "harness-home") : undefined, environment),
    serviceEndpoint: desktopServiceOrigin,
  }).pipe(Effect.provideService(CommandExecutor.CommandExecutor, executor))
}).pipe(Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer, BunSqliteDriverLayer)))
