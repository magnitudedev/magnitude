import { describe, expect, it } from "vitest"
import { mkdtemp, rm, stat } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { fileURLToPath } from "node:url"

const invoke = async (args: readonly string[], environment: Record<string, string> = {}) => {
  const process = Bun.spawn([Bun.which("bun")!, fileURLToPath(new URL("../index.ts", import.meta.url)), ...args], {
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
  it.each(["setup", "service", "--prompt", "--resume", "--system-override", "--atif"])("rejects removed interactive input %s", async (argument) => {
    const result = await invoke([argument])
    expect(result.code).not.toBe(0)
    expect(result.stderr).toContain(argument.startsWith("--") ? "unknown option" : "too many arguments")
    expect(result.stdout).toBe("")
  })
  it.each(["", "catalog", "models", "connections", "status", "serve", "docs", "update"])("supports finite help for %s", async (command) => {
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
  it.each([
    ["hardware"],
    ["catalog", "status"], ["catalog", "list"], ["catalog", "recommendations"],
    ["catalog", "show", "test:gguf:q4"], ["catalog", "pull", "test:gguf:q4"],
    ["catalog", "cancel", "test:gguf:q4"], ["catalog", "remove", "test:gguf:q4"],
    ["models", "status"], ["models", "status", "test:gguf:q4"],
    ["models", "load", "test:gguf:q4"], ["models", "stop"],
    ["connections", "add", "pi"], ["connections", "sync", "pi"],
  ])("%j requires an existing owner without creating a profile", async (...args) => {
    const root = await mkdtemp(join(tmpdir(), "mag-connect-only-"))
    const profile = join(root, "absent")
    try {
      const result = await invoke(args, { MAGNITUDE_DEV_DATA_DIR: profile, MAGNITUDE_DEV_PORT: "11168" })
      expect(result).toEqual({
        code: 1, stdout: "",
        stderr: "No Magnitude service is running. Open the Magnitude desktop app or run `magnitude serve`.\n",
      })
      await expect(stat(profile)).rejects.toMatchObject({ code: "ENOENT" })
    } finally { await rm(root, { recursive: true, force: true }) }
  })
  it("status reports absence successfully without creating a profile", async () => {
    const root = await mkdtemp(join(tmpdir(), "mag-status-"))
    const profile = join(root, "absent")
    try {
      const result = await invoke(["status"], { MAGNITUDE_DEV_DATA_DIR: profile })
      expect(result.code, result.stderr).toBe(0)
      expect(result.stderr).toBe("")
      expect(result.stdout).toContain("Runtime         Stopped")
      expect(result.stdout).toContain("Owner           None")
      expect(result.stdout).not.toMatch(/Tray|Starts at login/)
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
