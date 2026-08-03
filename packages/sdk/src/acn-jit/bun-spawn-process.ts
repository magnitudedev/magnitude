import { Effect, Option } from "effect"
import { DaemonSpawnFailed } from "./errors"
import type { ChildProcessSpawner } from "./local-daemon"
import { captureSpawnDiagnostics } from "./spawn-diagnostic"

const chunks = (
  stream: ReadableStream<Uint8Array>,
): AsyncIterable<Uint8Array> => ({
  async *[Symbol.asyncIterator]() {
    const reader = stream.getReader()
    try {
      while (true) {
        const result = await reader.read()
        if (result.done) return
        yield result.value
      }
    } finally {
      reader.releaseLock()
    }
  },
})

/**
 * Bun implementation for detached daemons.
 *
 * The detached process is unreferenced so it can outlive its launching client.
 * While the client remains alive, stdout and stderr are continuously drained
 * into a bounded diagnostic tail for startup failures.
 */
export const BunDetachedChildProcessSpawner: ChildProcessSpawner = {
  spawn: (command) =>
    Effect.try({
      try: () => {
        const process = Bun.spawn({
          cmd: Array.from(command),
          detached: true,
          stdio: ["ignore", "pipe", "pipe"],
          env: globalThis.process.env,
        })
        const diagnostics = captureSpawnDiagnostics([
          chunks(process.stdout),
          chunks(process.stderr),
        ])
        process.unref()
        return {
          pid: Option.some(process.pid),
          exited: Effect.promise(() => process.exited),
          diagnostic: diagnostics.diagnostic,
          kill: (signal) =>
            Effect.sync(() => {
              process.kill(signal)
            }),
        }
      },
      catch: (cause) =>
        new DaemonSpawnFailed({
          reason: `Failed to spawn Magnitude: ${String(cause)}`,
        }),
    }),
}
