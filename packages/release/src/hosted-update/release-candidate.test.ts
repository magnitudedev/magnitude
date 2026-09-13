import { Effect, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { ReleaseArtifactSchema } from "../contracts"
import { hostedDesktopManifests } from "./release-candidate"

const commit = "a".repeat(40)
const artifacts = ["darwin-arm64", "darwin-x64", "linux-arm64-gnu", "linux-x64-gnu"].flatMap(host =>
  (host.startsWith("darwin") ? ["dmg", "zip"] : ["deb", "rpm"]).map(format => Schema.decodeUnknownSync(ReleaseArtifactSchema)({
    id: format === "zip" ? `desktop-update-${host}` : format === "dmg" ? `desktop-${host}` : `desktop-${host}-${format}`,
    kind: "desktop", host, filename: `magnitude-${host}.${format}`, bytes: 123, sha256: "b".repeat(64),
  })))
const release = { version: "2.0.0", sourceCommit: commit, artifacts }
describe("production desktop publication admission", () => {
  it("preserves exact accepted byte identities across both Mac transports and both Linux formats", async () => {
    const result = await Effect.runPromise(hostedDesktopManifests(release, commit))
    expect(result).toHaveLength(8)
    expect(new Set(result.map(m => `${m.artifact.target.os}/${m.artifact.target.arch}/${m.artifact.target.package}`)).size).toBe(8)
    result.forEach((manifest, index) => expect(manifest.artifact).toMatchObject({ id: artifacts[index]!.id, bytes: 123, sha256: "b".repeat(64), path: `releases/2.0.0/${artifacts[index]!.filename}` }))
  })
  it.each([
    ["missing transport", artifacts.slice(1)],
    ["duplicate identity", [...artifacts.slice(1), artifacts[1]!]],
    ["mismatched host", [{ ...artifacts[0]!, host: artifacts[2]!.host }, ...artifacts.slice(1)]],
    ["wrong format", [{ ...artifacts[0]!, filename: "magnitude.zip" }, ...artifacts.slice(1)]],
    ["escaping filename", [{ ...artifacts[0]!, filename: "../magnitude.dmg" }, ...artifacts.slice(1)]],
  ])("rejects %s before publication", async (_, invalid) => {
    const result = await Effect.runPromise(hostedDesktopManifests({ ...release, artifacts: invalid }, commit).pipe(Effect.either))
    expect(result._tag).toBe("Left")
  })
  it("refuses a different released source", async () => {
    expect((await Effect.runPromise(hostedDesktopManifests(release, "c".repeat(40)).pipe(Effect.either)))._tag).toBe("Left")
  })
})
