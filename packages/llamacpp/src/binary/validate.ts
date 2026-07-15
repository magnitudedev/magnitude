import { Chunk, Effect, Stream } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import { parseVersionNumber } from "../version"
import { LlamaCppExecutableValidationError } from "../errors"

/**
 * Run `llama-server --version` and return the parsed build number.
 * Captures both stdout and stderr since Metal init logs may precede the version line.
 */
export function validateBinary(
  binaryPath: string,
): Effect.Effect<number, LlamaCppExecutableValidationError, CommandExecutor.CommandExecutor> {
  return Effect.scoped(Effect.gen(function* () {
    const process = yield* Command.start(Command.make(binaryPath, "--version")).pipe(
      Effect.mapError((cause) =>
        new LlamaCppExecutableValidationError({
          path: binaryPath,
          reason: `Failed to execute: ${cause}`,
          cause,
        }),
      ),
    )
    const collect = (stream: Stream.Stream<Uint8Array, unknown>) => stream.pipe(
      Stream.decodeText(),
      Stream.runCollect,
      Effect.map((chunks) => Chunk.toReadonlyArray(chunks).join("")),
    )
    const [stdout, stderr, exitCode] = yield* Effect.all([
      collect(process.stdout),
      collect(process.stderr),
      process.exitCode,
    ], { concurrency: 3 }).pipe(
      Effect.mapError((cause) => new LlamaCppExecutableValidationError({
        path: binaryPath,
        reason: `Failed to read version output: ${cause}`,
        cause,
      })),
    )
    if (Number(exitCode) !== 0) {
      return yield* new LlamaCppExecutableValidationError({
        path: binaryPath,
        reason: `llama-server --version exited with status ${exitCode}: ${(stderr || stdout).trim()}`,
      })
    }
    return yield* Effect.try({
      try: () => parseVersionNumber(`${stdout}\n${stderr}`),
      catch: (cause) => new LlamaCppExecutableValidationError({
        path: binaryPath,
        reason: cause instanceof Error ? cause.message : String(cause),
        cause,
      }),
    })
  }))
}
