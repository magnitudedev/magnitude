import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import { spawn } from "node:child_process"
import { describe, expect, it } from "vitest"
const addon = fileURLToPath(new URL(`../../dist/native/${process.platform}-${process.arch}/desktop-host.node`, import.meta.url))
const fixture = fileURLToPath(new URL("./fixtures/unix-continuation.cjs", import.meta.url))
describe.skipIf(process.platform === "win32")("Unix foreground continuation", () => {
  it("preserves PID, invocation, cwd, environment and streams while releasing ownership on exec", async () => {
    const directory = await mkdtemp(join(tmpdir(), "magnitude-continuation-"))
    const argument = "spaces ' $literal `literal`"
    try {
      const result = await new Promise<{ code: number | null; output: string; error: string }>((resolve, reject) => {
        const child = spawn(process.execPath, [fixture, addon, directory, argument], { cwd: directory,
          env: { ...process.env, CONTINUATION_FIXTURE: "preserved value" }, stdio: ["pipe", "pipe", "pipe"] })
        let output = "", error = ""
        child.stdout.on("data", bytes => { output += bytes })
        child.stderr.on("data", bytes => { error += bytes })
        child.on("error", reject)
        child.on("close", code => resolve({ code, output, error }))
        child.stdin.end("preserved stdin")
      })
      expect(result.code).toBe(0)
      expect(result.error).toBe("replacement stderr")
      const output = JSON.parse(result.output)
      expect(output.pid).toBe(output.previous)
      expect(output).toMatchObject({ cwd: directory.replace(/^\/var\//, "/private/var/"), argument, value: "preserved value", stdin: "preserved stdin" })
    } finally { await rm(directory, { recursive: true, force: true }) }
  })
  it("rejects malformed input and failed execution without terminating the caller", () => {
    const native = createRequire(import.meta.url)(addon) as { replaceProcess(path: string, args: unknown, environment: unknown): never }
    for (const [path, args, environment] of [["relative", [], []], ["/bin/sh\0bad", [], []], ["/bin/sh", ["bad\0value"], []],
      ["/bin/sh", [], ["invalid"]], ["/bin/sh", [], ["=invalid"]], ["/bin/sh", {}, []], ["/absent/magnitude", [], []]] as const) {
      expect(() => native.replaceProcess(path, args, environment)).toThrow()
    }
  })
})
