import { Effect } from "effect"
import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import { parseVersionNumber } from "../version"
import { LlamaCppBinaryValidationFailed } from "../errors"

/**
 * Run `llama-server --version` and return the parsed build number.
 * Captures both stdout and stderr since Metal init logs may precede the version line.
 */
export function validateBinary(
  binaryPath: string,
): Effect.Effect<number, LlamaCppBinaryValidationFailed, CommandExecutor.CommandExecutor> {
  return Effect.gen(function* () {
    const result = yield* Command.string(
      Command.make(binaryPath, "--version"),
    ).pipe(
      Effect.mapError((cause) =>
        new LlamaCppBinaryValidationFailed({
          path: binaryPath,
          reason: `Failed to execute: ${cause}`,
        }),
      ),
    )
    return parseVersionNumber(result)
  })
}
