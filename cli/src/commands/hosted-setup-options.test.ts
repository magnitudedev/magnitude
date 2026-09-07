import { spawnSync } from "node:child_process"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { Effect, Schema } from "effect"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { HostedSetupCapability, HostedSetupResult } from "@magnitudedev/client-common/harness-connections/hosted-setup"

// Exercise the actual lazy parser boundary, including capability mode without a TTY.
const run = (...args: string[]) => spawnSync("bun", [
  fileURLToPath(new URL("../index.tsx", import.meta.url)), "setup", ...args,
], { encoding: "utf8", timeout: 15_000 })

describe("hosted setup CLI options", () => {
  it("preserves ordinary setup help", () => {
    const result = run("--help")
    expect(result.status).toBe(0)
    expect(result.stdout).toContain("Interactive first time setup")
  })
  it("reports capability without an interactive terminal", () => {
    const result = run("--host-protocol")
    expect(result.status).toBe(0)
    expect(result.stderr).toBe("")
    expect(Schema.decodeUnknownSync(Schema.parseJson(HostedSetupCapability))(result.stdout)).toEqual({ protocolVersion: 1 })
  })
  it("writes a private, structured non-TTY failure and never overwrites an existing result", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-host-options-" })
      const path = `${directory}/result.json`
      const result = run("--host", "pi", "--result-file", path)
      expect(result.status).toBe(1)
      const encoded = yield* fs.readFileString(path)
      expect(yield* Schema.decodeUnknown(Schema.parseJson(HostedSetupResult))(encoded)).toEqual({
        _tag: "Failed", protocolVersion: 1, message: "Hosted setup requires an interactive terminal",
      })
      expect((yield* fs.stat(path)).mode & 0o077).toBe(0)
      const retry = run("--host", "pi", "--result-file", path)
      expect(retry.status).toBe(1)
      expect(retry.stderr).toContain("requires a new result file")
      expect(yield* fs.readFileString(path)).toBe(encoded)
    })).pipe(Effect.provide(BunContext.layer)))
  })
  it("rejects relative result paths", () => {
    const result = run("--host", "pi", "--result-file", "result.json")
    expect(result.status).toBe(1)
    expect(result.stderr).toContain("must be absolute")
  })
  it.skipIf(process.platform === "win32")("rejects a non-private result directory", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-host-permissions-" })
      yield* fs.chmod(directory, 0o755)
      const result = run("--host", "pi", "--result-file", `${directory}/result.json`)
      expect(result.status).toBe(1)
      expect(result.stderr).toContain("private directory")
      expect(yield* fs.exists(`${directory}/result.json`)).toBe(false)
    })).pipe(Effect.provide(BunContext.layer)))
  })
  it.each([
    ["--host", "pi"], ["--result-file", "/tmp/result.json"],
    ["--host", "hermes", "--result-file", "/tmp/result.json"],
    ["--host-protocol", "--host", "pi", "--result-file", "/tmp/result.json"],
    ["--host-protocol", "--prompt", "hello"],
    ["--host-protocol", "--resume"],
    ["--host-protocol", "--atif", "/tmp/file"],
    ["--host-protocol", "--system-override", "text"],
  ])("rejects incompatible options %j", (...args) => {
    const result = run(...args)
    expect(result.status).toBe(1)
    expect(result.stderr).toMatch(/Hosted setup|--host-protocol/)
    expect(result.stdout).toBe("")
  })
})
