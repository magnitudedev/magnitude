import { spawn } from "node:child_process"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

// Opt-in acceptance against the real bundle, with a disposable data/profile directory.
// Playwright's Electron driver requires Node. Bun still owns Vitest and ordinary package tests.
describe.skipIf(!process.env.MAGNITUDE_TEST_DESKTOP_EXECUTABLE || process.platform !== "darwin")("packaged application lifecycle", () => {
  it("preserves background intent, forwards Open, and retires the service on Quit", async () => {
    const result = await new Promise<{ code: number | null; output: string }>((resolve, reject) => {
      const child = spawn(process.env.MAGNITUDE_TEST_NODE ?? "node", [fileURLToPath(new URL("./fixtures/packaged-lifecycle.mjs", import.meta.url))], { env: process.env, stdio: ["ignore", "pipe", "pipe"] })
      let output = ""
      const collect = (chunk: Buffer) => { output = (output + chunk.toString()).slice(-32_000); console.log(chunk.toString().trimEnd()) }
      child.stdout.on("data", collect)
      child.stderr.on("data", collect)
      child.once("error", reject)
      child.once("exit", code => resolve({ code, output }))
    })
    expect(result.code, result.output).toBe(0)
  }, 120_000)
})
