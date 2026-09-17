import { spawn } from "node:child_process"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

// Run only against an explicitly installed package in an isolated Linux consumer.
// The fixture owns a disposable HOME and never invokes a package manager itself.
describe.skipIf(process.platform !== "linux" || process.env.MAGNITUDE_TEST_INSTALLED_LINUX !== "1")("installed Linux desktop", () => {
  it("preserves login state, hidden launch, canonical CLI discovery and full Quit", async () => {
    const node = process.env.MAGNITUDE_TEST_NODE
    if (!node) throw new Error("Set MAGNITUDE_TEST_NODE to the absolute Node executable; Bun's node shim cannot run Playwright Electron acceptance")
    const result = await new Promise<{ code: number | null; output: string }>((resolve, reject) => {
      const child = spawn(node, [fileURLToPath(new URL("./fixtures/linux-installed-lifecycle.mjs", import.meta.url))], {
        env: process.env, stdio: ["ignore", "pipe", "pipe"],
      })
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
