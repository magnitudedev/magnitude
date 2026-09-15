import { describe, expect, it } from "vitest"
import { run } from "./common"

describe("build commands", () => {
  it("drains both output streams while waiting for the command", async () => {
    const output = await run([process.execPath, "-e", "process.stderr.write('diagnostic'.repeat(16384)); process.stdout.write('complete')"])
    expect(output).toBe("complete")
  })

  it("reports a failed command instead of returning its output", async () => {
    await expect(run([process.execPath, "-e", "console.error('compile failed'); process.exit(7)"]))
      .rejects.toThrow("failed with exit 7: compile failed")
  })

  it("preserves compiler diagnostics written to stdout when stderr also has a summary", async () => {
    await expect(run([process.execPath, "-e", "console.log('linker diagnostic'); console.error('build failed'); process.exit(1)"]))
      .rejects.toThrow("linker diagnostic\nbuild failed")
  })
})
