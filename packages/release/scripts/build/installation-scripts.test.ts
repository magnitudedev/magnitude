import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { execFileSync } from "node:child_process"
import { generateKeyPairSync } from "node:crypto"
import { describe, expect, it } from "vitest"
import { renderUnixInstallationScript, renderWindowsInstallationScript } from "./installation-scripts"

const publicKey = generateKeyPairSync("ed25519").publicKey.export({ type: "spki", format: "pem" }).toString()
const render = (origin = "https://magnitude.dev", appleTeam = "ABCDEFGHIJ") =>
  renderUnixInstallationScript({ origin, appleTeam, publicKey }).pipe(Effect.provide(NodeContext.layer))

describe("installation script generation", () => {
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
