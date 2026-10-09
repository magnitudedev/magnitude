import { Context, Effect, Layer, Schema } from "effect"
import { spawn } from "node:child_process"

export class TerminalCommandFailed extends Schema.TaggedError<TerminalCommandFailed>()("TerminalCommandFailed", { message: Schema.String }) {}

/**
 * Runs a command in the caller's session, attached to its controlling terminal, and returns its exit
 * code. The platform executor starts Unix children in a new session, where sudo can neither reuse the
 * terminal's ticket nor ask for a password. With `stdin: "lifetime"` the child reads a pipe that
 * closes only when this process ends; output always goes to the caller's terminal.
 */
export interface TerminalCommand {
  readonly run: (executable: string, args: readonly string[], options: { readonly stdin: "inherit" | "lifetime" }) =>
    Effect.Effect<number, TerminalCommandFailed>
}
export const TerminalCommand = Context.GenericTag<TerminalCommand>("daemon-management/TerminalCommand")

export const nodeTerminalCommand = Layer.succeed(TerminalCommand, TerminalCommand.of({
  run: (executable, args, options) => Effect.async<number, TerminalCommandFailed>(resume => {
    const child = spawn(executable, [...args], { stdio: [options.stdin === "inherit" ? "inherit" : "pipe", "inherit", "inherit"] })
    child.once("error", error => resume(Effect.fail(new TerminalCommandFailed({ message: error.message }))))
    child.once("exit", (code, signal) => resume(Effect.succeed(code ?? (signal ? 128 : 1))))
    return Effect.sync(() => { if (child.exitCode === null && child.signalCode === null) child.kill("SIGTERM") })
  }),
}))
