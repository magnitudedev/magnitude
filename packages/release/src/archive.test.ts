import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Option } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import { buildArchive } from "../scripts/build/common"
import { ArchiveExtractor, NodeArchiveExtractor } from "./archive"
import type { HostId } from "./targets"

const roots: string[] = []
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})

/** Packs `paths` as the host's inference artifact and extracts it through the client extractor. */
const extract = async (host: HostId, paths: readonly string[]) => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-archive-test-"))
  roots.push(root)
  const source = join(root, "payload")
  await writeFile(source, "payload")
  const artifact = await buildArchive(
    join(root, "artifact.tar.gz"),
    join(root, "artifact.json"),
    { id: `icn-base-${host}`, kind: "icn-base", host: Option.some(host), nativeBuild: Option.some("engine-build") },
    paths.map((path) => ({ path, source, mode: 0o644 })),
  )
  return Effect.runPromise(Effect.gen(function* () {
    const extractor = yield* ArchiveExtractor
    return yield* extractor.extract(join(root, "artifact.tar.gz"), join(root, "installed"), artifact, Option.none())
  }).pipe(Effect.provide(NodeArchiveExtractor), Effect.either))
}

const linuxNvrtc = ["runtime/libnvrtc.so.12", "runtime/libnvrtc-builtins.so.12.9", "runtime/NVRTC-LICENSE.txt"]
const layout = ["bin/magnitude-inference", "catalog/model-planner-inputs.bundle"]

describe("inference artifact layout", () => {
  it("accepts a Linux artifact carrying NVRTC in runtime/", async () => {
    expect((await extract("linux-x64-gnu", [...layout, ...linuxNvrtc]))._tag).toBe("Right")
  })

  it("requires NVRTC on CUDA hosts", async () => {
    const result = await extract("linux-arm64-gnu", [...layout, "runtime/libnvrtc.so.12", "runtime/NVRTC-LICENSE.txt"])
    expect(result._tag === "Left" && result.left.message).toContain("runtime/libnvrtc-builtins.so.12.9")
  })

  it("accepts an Apple artifact without a runtime directory", async () => {
    expect((await extract("darwin-arm64", layout))._tag).toBe("Right")
  })

  it("rejects backend modules; there are no backend packs", async () => {
    const result = await extract("darwin-arm64", [...layout, "backends/libggml-cpu.so"])
    expect(result._tag === "Left" && result.left.message).toContain("unexpected path")
  })
})
