import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import releasePlan from "../release-plan.json"
import { ReleaseManifestSchema, releaseTag } from "./contracts"

const version = "1.2.3"

const manifest = Schema.decodeUnknownSync(ReleaseManifestSchema)({
  schemaVersion: 2,
  version,
  acnRevision: 1,
  rpc: releasePlan.rpc,
  plugins: [],
  tag: releaseTag(version),
  sourceCommit: "a".repeat(40),
  artifacts: [
    { id: "desktop-darwin-arm64", kind: "desktop", host: "darwin-arm64", filename: "desktop.zip", bytes: 1, sha256: "b".repeat(64) },
    { id: "icn-base-darwin-arm64", kind: "icn-base", host: "darwin-arm64", filename: "inference.tar.gz", bytes: 1, sha256: "c".repeat(64), nativeBuild: "engine-1" },
  ],
})

describe("release manifest wire shape", () => {
  it("keeps what shipped desktops require to decode an update", () => {
    const encoded = Schema.encodeSync(ReleaseManifestSchema)(manifest)
    expect(encoded.schemaVersion).toBe(2)
    const [desktop, inference] = encoded.artifacts
    expect(desktop).not.toHaveProperty("backend")
    expect(desktop).not.toHaveProperty("backendModuleAbi")
    expect(inference).toMatchObject({ kind: "icn-base", backend: "cpu", nativeBuild: "engine-1" })
    expect(inference?.backendModuleAbi).toEqual(expect.any(String))
  })

  it("decodes back to the artifact without the compatibility fields", () => {
    const decoded = Schema.decodeUnknownSync(ReleaseManifestSchema)(Schema.encodeSync(ReleaseManifestSchema)(manifest))
    expect(decoded.artifacts[1]).not.toHaveProperty("backend")
    expect(decoded.artifacts[1]?.nativeBuild).toEqual(Option.some("engine-1"))
  })
})
