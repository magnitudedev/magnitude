import { Effect, Context, Option } from "effect"
import { DaemonError } from "./errors"
import type { JitDaemonProvider } from "../jit-rpc"

/**
 * Spawner abstraction for daemon lifecycle.
 *
 * The spawner is the single variable point across environments:
 * - **Local (Bun/Node):** reads registration files, health-checks, spawns
 *   processes directly via `makeLocalDaemonSpawner`.
 * - **Remote (Browser):** delegates `discover`/`spawn` to a proxy server via
 *   HTTP fetch via `makeRemoteDaemonSpawner`.
 *
 * Both `discover` and `spawn` require `never` — all dependencies
 * (`FileSystem`, `HttpClient`, `CommandExecutor`) are captured at
 * construction time (see `makeLocalDaemonSpawner` /
 * `makeRemoteDaemonSpawner`) and sealed inside the returned spawner.
 */
export interface DaemonSpawner {
  /** Discover a healthy daemon URL. Returns `None` if none found. */
  readonly discover: () => Effect.Effect<Option.Option<string>, DaemonError, never>
  /**
   * Spawn a daemon process and wait until it is healthy.
   * Returns the URL of the now-healthy daemon.
   * If `command` is `undefined`, the spawner resolves the binary itself
   * (local spawners use `resolveBinaryCommand`; remote spawners delegate to
   * the proxy's default).
   */
  readonly spawn: (command: string[] | undefined) => Effect.Effect<string, DaemonError, never>
}

export class DaemonSpawnerTag extends Context.Tag("DaemonSpawner")<
  DaemonSpawnerTag,
  DaemonSpawner
>() {}

export const toJitDaemonProvider = (
  spawner: DaemonSpawner,
  spawnCommand?: string[],
): JitDaemonProvider<DaemonError> => ({
  discover: () =>
    spawner.discover().pipe(
      Effect.map(Option.map((url) => ({ url }))),
    ),
  spawn: () =>
    spawner.spawn(spawnCommand).pipe(
      Effect.map((url) => ({ url })),
    ),
})
