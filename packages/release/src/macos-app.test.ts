import { Effect, Option, Schema } from "effect"
import { mkdtemp, mkdir, readFile, rm, stat } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { createWriteStream } from "node:fs"
import { pipeline } from "node:stream/promises"
import { createGzip } from "node:zlib"
import { pack } from "tar-stream"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { ArchiveExtractor, NodeArchiveExtractor } from "./archive"
import { ReleaseArtifactSchema } from "./contracts"
import { MACOS_REQUIRED_FILES } from "./macos-app"

let root: string
beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "magnitude-app-archive-")) })
afterEach(() => rm(root, { recursive: true, force: true }))
const artifact = Schema.decodeUnknownSync(ReleaseArtifactSchema)({ id: "acn-darwin-arm64", kind: "acn", host: "darwin-arm64", filename: "app.tar.gz", bytes: 1, sha256: "a".repeat(64) })
const makeArchive = async (entries: readonly { name: string; type?: "file" | "symlink" }[]) => {
  const tar = pack()
  const done = pipeline(tar, createGzip(), createWriteStream(join(root, "app.tar.gz")))
  for (const entry of entries) tar.entry({ ...entry, type: entry.type ?? "file", mode: entry.name.includes("/MacOS/") ? 0o755 : 0o644, ...(entry.type === "symlink" ? { linkname: "../../outside" } : {}) }, entry.type === "symlink" ? undefined : Buffer.from(entry.name))
  tar.finalize()
  await done
  await mkdir(join(root, "extracted"))
  return Effect.runPromise(Effect.gen(function* () {
    const extractor = yield* ArchiveExtractor
    yield* extractor.extract(join(root, "app.tar.gz"), join(root, "extracted"), artifact, Option.none())
  }).pipe(Effect.provide(NodeArchiveExtractor)))
}

describe("Magnitude app distribution", () => {
  it("does not treat a desktop installer as a runtime archive", async () => {
    const desktop = Schema.decodeUnknownSync(ReleaseArtifactSchema)({ id: "desktop-darwin-arm64", kind: "desktop", host: "darwin-arm64", filename: "Magnitude.dmg", bytes: 1, sha256: "a".repeat(64) })
    const result = await Effect.runPromise(Effect.gen(function* () {
      const extractor = yield* ArchiveExtractor
      return yield* extractor.extract(join(root, "not-opened.dmg"), join(root, "not-created"), desktop, Option.none()).pipe(Effect.either)
    }).pipe(Effect.provide(NodeArchiveExtractor)))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toContain("platform installer")
    expect(await stat(join(root, "not-created")).then(() => true, () => false)).toBe(false)
  })
  it("preserves every sealed resource, ticket file, and executable mode through the production extractor", async () => {
    const names = [...MACOS_REQUIRED_FILES.map((name) => `Magnitude.app/${name}`), "Magnitude.app/Contents/CodeResources"]
    await makeArchive(names.map((name) => ({ name })))
    for (const name of names) expect(await readFile(join(root, "extracted", name), "utf8")).toBe(name)
    expect((await stat(join(root, "extracted/Magnitude.app/Contents/MacOS/magnitude-service"))).mode & 0o777).toBe(0o755)
  })
  it.each([
    [[{ name: "../escape" }]],
    [[{ name: "Magnitude.app/Contents/link", type: "symlink" as const }]],
    [[{ name: "Magnitude.app/Contents/A" }, { name: "Magnitude.app/Contents/a" }]],
    [[{ name: "Magnitude.app/Contents/é" }, { name: "Magnitude.app/Contents/é" }]],
    [[{ name: "bin/magnitude-service" }]],
  ])("rejects unsafe or obsolete app archives: %j", async (entries) => {
    await expect(makeArchive(entries)).rejects.toThrow()
  })
})
