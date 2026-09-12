import { spawn } from "node:child_process"
import type { Duplex, Readable } from "node:stream"
import { FSM } from "@magnitudedev/utils"
import { Context, Effect, Layer, Option, Schema } from "effect"

export class GuardedCommandFailed extends Schema.TaggedError<GuardedCommandFailed>()("GuardedCommandFailed", { message: Schema.String }) {}
export const GuardedCommandResult = Schema.Struct({ code: Schema.Int, stdout: Schema.String, stderr: Schema.String })
export interface GuardedCommand {
  readonly run: (executable: string, args: readonly string[], environment: Readonly<Record<string, string>>) => Effect.Effect<typeof GuardedCommandResult.Type, GuardedCommandFailed>
}
export const GuardedCommand = Context.GenericTag<GuardedCommand>("@magnitudedev/daemon-management/GuardedCommand")

class Running extends Schema.TaggedClass<Running>()("Running", {}) {}
class Retiring extends Schema.TaggedClass<Retiring>()("Retiring", {
  error: Schema.optionalWith(GuardedCommandFailed, { as: "Option", exact: true }),
}) {}
class Closed extends Schema.TaggedClass<Closed>()("Closed", {}) {}
const lifecycle = FSM.defineFSM({ Running, Retiring, Closed }, {
  Running: ["Retiring"], Retiring: ["Closed"], Closed: [],
} as const)

/** The native helper owns the process group even if this JavaScript process disappears. */
export const guardedCommandLayer = (helper: string) => Layer.succeed(GuardedCommand, {
  run: (executable, args, environment) => Effect.async(resume => {
    const child = spawn(helper, [executable, ...args], { env: environment, detached: true, stdio: ["ignore", "pipe", "pipe", "pipe", "pipe"] })
    const lifetime = child.stdio[3] as Duplex
    const buffers = [Buffer.alloc(0), Buffer.alloc(0), Buffer.alloc(0)]
    let ended = 0
    let state: Running | Retiring | Closed = new Running({})
    const retire = (error = Option.none<GuardedCommandFailed>()) => {
      if (state._tag === "Closed") return
      state = state._tag === "Running" ? lifecycle.transition(state, "Retiring", { error })
        : lifecycle.hold(state, { error: Option.orElse(state.error, () => error) })
      lifetime.destroy()
    }
    const fail = (message: string) => retire(Option.some(new GuardedCommandFailed({ message })))
    child.once("error", () => fail("Could not start the protected command helper."))
    lifetime.on("error", () => fail("Protected command lifetime channel failed."))
    ;[child.stdout!, child.stderr!, child.stdio[4] as Readable].forEach((stream, index) => {
      stream.on("data", (chunk: Buffer) => {
        if (buffers[index]!.length + chunk.length > (index === 2 ? 16 : 1024 * 1024)) return fail("Protected command output exceeded its limit.")
        buffers[index] = Buffer.concat([buffers[index]!, chunk])
      })
      stream.on("error", () => fail("Could not read protected command output."))
      stream.once("end", () => { if (++ended === 3) retire() })
    })
    child.once("exit", () => { if (state._tag === "Running") fail("Protected command helper exited before completion.") })
    child.once("close", () => {
      if (state._tag === "Running") fail("Protected command closed before completion.")
      if (state._tag !== "Retiring") return
      const error = state.error
      state = lifecycle.transition(state, "Closed", {})
      if (Option.isSome(error)) return resume(Effect.fail(error.value))
      const status = buffers[2]!.toString("utf8")
      if (!/^\d{1,3}\n$/.test(status) || Number(status) > 255) return resume(Effect.fail(new GuardedCommandFailed({ message: "Protected command returned an invalid exit status." })))
      resume(Effect.succeed({ code: Number(status), stdout: buffers[0]!.toString("utf8"), stderr: buffers[1]!.toString("utf8") }))
    })
    return Effect.async<void>(done => {
      if (state._tag === "Closed") return done(Effect.void)
      child.once("close", () => done(Effect.void))
      retire()
    })
  }),
})
