import { describe, expect, it } from "vitest"
import { Effect, Exit, Fiber } from "effect"
import { mkdtempSync, readFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { nodeTerminalCommand, TerminalCommand } from "./terminal-command"

const runInTerminal = (...args: Parameters<TerminalCommand["run"]>) => TerminalCommand.pipe(Effect.flatMap(terminal => terminal.run(...args)), Effect.provide(nodeTerminalCommand))

describe.skipIf(process.platform === "win32")("nodeTerminalCommand", () => {
  it("keeps the caller's session, so sudo can use the terminal", async () => {
    const out = join(mkdtempSync(join(tmpdir(), "terminal-command-")), "sid")
    // Field 6 of /proc/<pid>/stat (or ps on macOS) is the session id; it must match ours.
    const code = await Effect.runPromise(runInTerminal("/bin/sh", ["-c", `ps -o sess= -p $$ > ${out}`], { stdin: "inherit" }))
    expect(code).toBe(0)
    const own = (await Effect.runPromise(Effect.promise(() => import("node:child_process").then(cp =>
      cp.execFileSync("ps", ["-o", "sess=", "-p", String(process.pid)]).toString())))).trim()
    expect(readFileSync(out, "utf8").trim()).toBe(own)
  })

  it("returns the exit code and gives a lifetime stdin that stays open", async () => {
    expect(await Effect.runPromise(runInTerminal("/bin/sh", ["-c", "exit 3"], { stdin: "inherit" }))).toBe(3)
    // read -t returns nonzero on timeout (pipe open, no data) and zero on EOF.
    expect(await Effect.runPromise(runInTerminal("/bin/bash", ["-c", "read -t 1 line; [ $? -gt 128 ]"], { stdin: "lifetime" }))).toBe(0)
  })

  it("stops the child when interrupted", async () => {
    const exit = await Effect.runPromise(Effect.gen(function* () {
      const fiber = yield* Effect.fork(runInTerminal("/bin/sleep", ["30"], { stdin: "inherit" }))
      yield* Effect.sleep("200 millis")
      return yield* Fiber.interrupt(fiber)
    }).pipe(Effect.timeout("5 seconds")))
    expect(Exit.isInterrupted(exit)).toBe(true)
  })

  it("fails for a missing executable", async () => {
    const exit = await Effect.runPromiseExit(runInTerminal("/nonexistent/command", [], { stdin: "inherit" }))
    expect(Exit.isFailure(exit)).toBe(true)
  })
})
