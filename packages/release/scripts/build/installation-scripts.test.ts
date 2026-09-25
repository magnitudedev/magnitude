import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { execFileSync, spawnSync } from "node:child_process"
import { generateKeyPairSync } from "node:crypto"
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, realpathSync, symlinkSync, rmSync, existsSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { describe, expect, it } from "vitest"
import { renderUnixInstallationScript, renderWindowsInstallationScript } from "./installation-scripts"

const publicKey = generateKeyPairSync("ed25519").publicKey.export({ type: "spki", format: "pem" }).toString()
const render = (origin = "https://magnitude.dev", appleTeam = "ABCDEFGHIJ") =>
  renderUnixInstallationScript({ origin, appleTeam, publicKey }).pipe(Effect.provide(NodeContext.layer))

describe("installation script generation", () => {
  it.skipIf(process.platform !== "darwin")("constructs the Mac installer request with native plist tools", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "magnitude request ' "))
    try {
      const script = await Effect.runPromise(render())
      const start = script.indexOf("    /usr/bin/plutil -create ")
      const end = script.indexOf('    "$app/Contents/Resources/magnitude"', start)
      expect(start).toBeGreaterThan(0)
      expect(end).toBeGreaterThan(start)
      const offer = { release: { version: "1.2.3", bytes: 123, sha256: "abc", signature: "proof" }, download: "https://github.com/magnitudedev/magnitude/releases/download/v1.2.3/app.zip" }
      writeFileSync(join(scratch, "offer.json"), JSON.stringify(offer))
      const destination = "/Applications/Magnitude user's test.app"
      const result = spawnSync("/bin/sh", ["-eu", "-c", script.slice(start, end)], {
        encoding: "utf8", timeout: 15000, env: { ...process.env, scratch, destination, channel: "stable" },
      })
      expect(result.status, result.stderr).toBe(0)
      expect(JSON.parse(readFileSync(join(scratch, "request.json"), "utf8"))).toEqual({
        bundle: destination, archive: join(scratch, "magnitude.zip"), channel: "stable", offer,
      })
    } finally { rmSync(scratch, { recursive: true, force: true }) }
  })
  it("canonicalizes a linked temporary directory with a trailing slash before download", async () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude script ' "))
    try {
      const bin = join(root, "bin"), temporary = join(root, "temporary"), alias = join(root, "alias"), receipt = join(root, "curl-arguments")
      mkdirSync(bin); mkdirSync(temporary); symlinkSync(temporary, alias)
      writeFileSync(join(bin, "curl"), '#!/bin/sh\nprintf "%s\\n" "$@" > "$TEST_CURL_ARGUMENTS"\nexit 73\n', { mode: 0o700 })
      const script = await Effect.runPromise(render())
      const result = spawnSync("/bin/sh", ["-s"], { input: script, encoding: "utf8", timeout: 15000,
        env: { ...process.env, PATH: `${bin}:${process.env.PATH}`, TMPDIR: `${alias}/`, TEST_CURL_ARGUMENTS: receipt } })
      expect(result.status, result.stderr).toBe(73)
      const args = readFileSync(receipt, "utf8").trimEnd().split("\n")
      const output = args[args.indexOf("--output") + 1]!
      expect(output).toBe(resolve(output))
      expect(output.startsWith(`${realpathSync(temporary)}/magnitude-install.`)).toBe(true)
      expect(existsSync(dirname(output))).toBe(false)
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
  it("quotes the Windows publisher as literal PowerShell text", async () => {
    const script = await Effect.runPromise(renderWindowsInstallationScript({ origin: "https://magnitude.dev", publisher: "Publisher's Name" }).pipe(Effect.provide(NodeContext.layer)))
    expect(script).toContain("$publisher = 'Publisher''s Name'")
    expect(script).not.toMatch(/@MAGNITUDE_[A-Z_]+@/)
  })
  it("pins publisher configuration and emits valid shell syntax", async () => {
    const script = await Effect.runPromise(render())
    expect(script).not.toContain("@MAGNITUDE_")
    expect(script).toContain("origin='https://magnitude.dev'")
    expect(() => execFileSync("/bin/sh", ["-n"], { input: script })).not.toThrow()
  })
  it.each(["http://magnitude.dev", "https://magnitude.dev/path", "https://magnitude.dev';exit 0", "https://user@magnitude.dev"])(
    "rejects an unsafe origin before rendering", async origin => {
      expect(await Effect.runPromise(render(origin).pipe(Effect.isFailure))).toBe(true)
    })
  it("requires a concrete publisher identity", async () => {
    expect(await Effect.runPromise(render("https://magnitude.dev", "").pipe(Effect.isFailure))).toBe(true)
  })
})
