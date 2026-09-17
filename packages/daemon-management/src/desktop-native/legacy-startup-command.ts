import { execFile } from "node:child_process"
import { Context, Effect, Layer, Schema } from "effect"

export class LegacyStartupFailed extends Schema.TaggedError<LegacyStartupFailed>()("LegacyStartupFailed", { message: Schema.String }) {}
export const CommandResult = Schema.Struct({ code: Schema.Int, stdout: Schema.String, stderr: Schema.String })
export interface LegacyStartupCommands {
  readonly run: (executable: string, args: readonly string[]) => Effect.Effect<typeof CommandResult.Type, LegacyStartupFailed>
}
export const LegacyStartupCommands = Context.GenericTag<LegacyStartupCommands>("@magnitudedev/daemon-management/LegacyStartupCommands")
export const NativeLegacyStartupCommands = Layer.succeed(LegacyStartupCommands, {
  run: (executable, args) => Effect.async(resume => {
    const child = execFile(executable, [...args], { encoding: "utf8", timeout: 5000, maxBuffer: 1024 * 1024 }, (error, stdout, stderr) => {
      if (error !== null && typeof error.code !== "number") return resume(Effect.fail(new LegacyStartupFailed({ message: `Could not execute ${executable}: ${error.message}` })))
      resume(Effect.succeed({ code: typeof error?.code === "number" ? error.code : 0, stdout, stderr }))
    })
    return Effect.sync(() => child.kill())
  }),
})
