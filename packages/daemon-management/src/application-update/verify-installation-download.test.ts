import { FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { signUpdateRelease } from "../../../release/src/hosted-update/release"
import { verifyWindowsInstallationDownload } from "./verify-installation-download"

describe("Windows bootstrap release verification", () => {
  it.each(["valid", "signature", "modified", "truncated", "oversized", "architecture"])("checks %s input before installation", scenario =>
    Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude bootstrap ' " })
      const keys = yield* Effect.sync(() => generateKeyPairSync("ed25519"))
      const contents = Buffer.from("signed installer fixture")
      const target = scenario === "architecture" ? { os: "darwin", arch: "arm64", package: "mac-zip" } as const
        : { os: "windows", arch: "x64", package: "windows-exe" } as const
      const release = yield* signUpdateRelease({ version: "0.1.6", bytes: contents.length,
        sha256: createHash("sha256").update(contents).digest("hex") }, target, keys.privateKey)
      const offer = join(root, "offer.json"), artifact = join(root, "installer.exe")
      yield* fs.writeFileString(offer, JSON.stringify({ release: scenario === "signature" ? { ...release, signature: "A".repeat(86) + "==" } : release,
        download: "https://github.com/magnitudedev/magnitude/releases/download/test/installer.exe" }))
      yield* fs.writeFile(artifact, scenario === "modified" ? Buffer.alloc(contents.length) : scenario === "truncated" ? contents.subarray(1)
        : scenario === "oversized" ? Buffer.concat([contents, contents]) : contents)
      expect(yield* verifyWindowsInstallationDownload({ offer, artifact, channel: "stable",
        publicKey: keys.publicKey.export({ type: "spki", format: "pem" }).toString() }).pipe(Effect.isSuccess)).toBe(scenario === "valid")
    })).pipe(Effect.provide(NodeContext.layer))))
})
