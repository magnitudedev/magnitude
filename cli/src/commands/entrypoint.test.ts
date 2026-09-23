import { describe, expect, it } from "vitest"
import { mkdtemp, rm, stat } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"

const invoke = async (args: readonly string[], environment: Record<string, string> = {}) => {
  const process = Bun.spawn([Bun.which("bun")!, new URL("../index.ts", import.meta.url).pathname, ...args], {
    stdin: "ignore", stdout: "pipe", stderr: "pipe",
    env: { ...globalThis.process.env, ...environment },
  })
  try {
    const [code, stdout, stderr] = await Promise.all([process.exited, new Response(process.stdout).text(), new Response(process.stderr).text()])
    return { code, stdout, stderr }
  } finally {
    process.kill()
  }
}

describe("headless CLI entrypoint", () => {
  it("prints help and exits without a terminal", async () => {
    const result = await invoke([])
    expect(result.code).toBe(0)
    expect(result.stdout).toContain("Usage: magnitude")
    expect(result.stdout).toContain("catalog")
    expect(result.stdout).not.toMatch(/\x1b\[/)
    expect(result.stderr).toBe("")
  })
  it.each(["setup", "--prompt", "--resume", "--system-override", "--atif"])("rejects removed interactive input %s", async (argument) => {
    const result = await invoke([argument])
    expect(result.code).not.toBe(0)
    expect(result.stderr).toContain("error:")
    expect(result.stdout).toBe("")
  })
  it.each(["", "catalog", "models", "connections", "service", "serve", "docs", "update"])("supports finite help for %s", async (command) => {
    const result = await invoke([...(command ? [command] : []), "--help"])
    expect(result.code).toBe(0)
    expect(result.stdout).toContain("Usage:")
    expect(result.stderr).toBe("")
  })
  it("serve help and rejected configuration flags never create ownership state", async () => {
    const root = await mkdtemp(join(tmpdir(), "mag-help-"))
    const profile = join(root, "unused")
    try {
      const env = { MAGNITUDE_DEV_DATA_DIR: profile }
      expect((await invoke(["serve", "--help"], env)).code).toBe(0)
      for (const flag of ["--port", "--data-dir", "--host"]) {
        const result = await invoke(["serve", flag, "invalid"], env)
        expect(result.code).toBe(1)
        expect(result.stderr).toContain("unknown option")
      }
      await expect(stat(profile)).rejects.toMatchObject({ code: "ENOENT" })
    } finally { await rm(root, { recursive: true, force: true }) }
  })
  it("prints only the version", async () => {
    const result = await invoke(["--version"])
    expect(result.code).toBe(0)
    expect(result.stdout.trim()).toMatch(/^\d+\.\d+\.\d+/)
    expect(result.stdout.trim().split("\n")).toHaveLength(1)
    expect(result.stderr).toBe("")
  })
})
