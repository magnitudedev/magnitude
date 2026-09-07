import { spawnSync } from "node:child_process"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const run = (...args: string[]) => spawnSync("bun", [
  fileURLToPath(new URL("../index.tsx", import.meta.url)), ...args,
], { encoding: "utf8", timeout: 15_000 })

describe("private Pi setup command", () => {
  it("preserves ordinary setup help", () => {
    const result = run("setup", "--help")
    expect(result.status).toBe(0)
    expect(result.stdout).toContain("Interactive first time setup")
    expect(result.stdout).not.toMatch(/host|result-file|setup-pi/)
  })
  it("does not advertise the internal entrypoint in root help", () => {
    const result = run("--help")
    expect(result.status).toBe(0)
    expect(result.stdout).not.toContain("setup-pi")
    expect(result.stderr).toBe("")
  })
  it("keeps version reporting independent of interactive startup", () => {
    const result = run("--version")
    expect(result.status).toBe(0)
    expect(result.stdout.trim()).toMatch(/^\d+\.\d+\.\d+/)
    expect(result.stderr).toBe("")
  })
  it("rejects non-TTY setup without a file or service request", () => {
    const result = run("setup-pi")
    expect(result.status).toBe(1)
    expect(result.stdout).toBe("")
    expect(result.stderr.trim()).toBe("Pi setup requires an interactive terminal")
  })
  it.each([
    ["setup", "--host", "pi"], ["setup", "--result-file", "/tmp/result.json"],
    ["setup", "--host-protocol"], ["setup-pi", "/tmp/result.json"],
    ["setup-pi", "--prompt", "hello"], ["setup-pi", "--resume"],
    ["setup-pi", "--atif", "/tmp/file"], ["setup-pi", "--system-override", "text"],
    ["--prompt", "hello", "setup-pi"], ["--resume", "session-id", "setup-pi"],
  ])("rejects incompatible options %j", (...args) => {
    const result = run(...args)
    expect(result.status).toBe(1)
    expect(result.stderr).toMatch(/Pi setup|unknown option|too many arguments/)
    expect(result.stdout).toBe("")
  })
})
