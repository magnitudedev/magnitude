import { BunContext } from "@effect/platform-bun"
import { createHash } from "node:crypto"
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { basename, join } from "node:path"
import { Effect, Option } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import { hostById, releaseHosts, type NvrtcRedistributable } from "../../src/targets"
import { run } from "./common"
import { NvrtcStagingFailed, stageNvrtc } from "./nvrtc"

const roots: string[] = []
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })))
})

/** A miniature NVIDIA archive: both libraries, the `.alt` variant that must not ship, and the license. */
const fixtureArchive = async () => {
  const root = await mkdtemp(join(tmpdir(), "magnitude-nvrtc-test-"))
  roots.push(root)
  const archiveRoot = "cuda_nvrtc-linux-x86_64-12.9.86-archive"
  await mkdir(join(root, archiveRoot, "lib"), { recursive: true })
  await writeFile(join(root, archiveRoot, "lib", "libnvrtc.so.12.9.86"), "nvrtc")
  await writeFile(join(root, archiveRoot, "lib", "libnvrtc-builtins.so.12.9.86"), "builtins")
  await writeFile(join(root, archiveRoot, "lib", "libnvrtc.alt.so.12.9.86"), "alt")
  await writeFile(join(root, archiveRoot, "LICENSE"), "license")
  const archive = join(root, `${archiveRoot}.tar.xz`)
  await run(["tar", "-cJf", archive, "-C", root, archiveRoot])
  const bytes = await readFile(archive)
  return { root, archiveRoot, archive, bytes, sha256: createHash("sha256").update(bytes).digest("hex") }
}

const serve = (archive: string) => {
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: () => new Response(Bun.file(archive)) })
  return { server, url: `http://127.0.0.1:${server.port}/cuda_nvrtc.tar.xz` }
}

const pin = (url: string, bytes: number, sha256: string, archiveRoot: string): NvrtcRedistributable => ({
  version: "12.9.86",
  url,
  bytes,
  sha256,
  libraries: [
    { member: `${archiveRoot}/lib/libnvrtc.so.12.9.86`, name: "libnvrtc.so.12" },
    { member: `${archiveRoot}/lib/libnvrtc-builtins.so.12.9.86`, name: "libnvrtc-builtins.so.12.9" },
  ],
  license: { member: `${archiveRoot}/LICENSE`, name: "NVRTC-LICENSE.txt" },
})

describe("NVRTC staging", () => {
  it("verifies the pinned archive and stages only the two libraries and the license under their installed names", async () => {
    const fixture = await fixtureArchive()
    const { server, url } = serve(fixture.archive)
    try {
      const cache = join(fixture.root, "cache")
      const redistributable = pin(url, fixture.bytes.byteLength, fixture.sha256, fixture.archiveRoot)
      const files = await Effect.runPromise(stageNvrtc(redistributable, cache).pipe(Effect.provide(BunContext.layer)))
      const staged = [...files.libraries, files.license]
      expect(staged.map((file) => basename(file))).toEqual(["libnvrtc.so.12", "libnvrtc-builtins.so.12.9", "NVRTC-LICENSE.txt"])
      expect(await Promise.all(staged.map((file) => readFile(file, "utf8")))).toEqual(["nvrtc", "builtins", "license"])
      expect((await readdir(join(cache, fixture.sha256))).sort()).toEqual(["NVRTC-LICENSE.txt", "libnvrtc-builtins.so.12.9", "libnvrtc.so.12"].sort())
      expect(await readdir(cache)).toEqual([fixture.sha256])

      server.stop(true)
      const cached = await Effect.runPromise(stageNvrtc(redistributable, cache).pipe(Effect.provide(BunContext.layer)))
      expect(cached).toEqual(files)
    } finally {
      server.stop(true)
    }
  })

  it("rejects an archive whose digest differs from the pin and publishes nothing", async () => {
    const fixture = await fixtureArchive()
    const { server, url } = serve(fixture.archive)
    try {
      const cache = join(fixture.root, "cache")
      const error = await Effect.runPromise(stageNvrtc(
        pin(url, fixture.bytes.byteLength, "0".repeat(64), fixture.archiveRoot),
        cache,
      ).pipe(Effect.provide(BunContext.layer), Effect.flip))
      expect(error).toBeInstanceOf(NvrtcStagingFailed)
      expect(await readdir(cache)).toEqual([])
    } finally {
      server.stop(true)
    }
  })

  it("repairs cached staging when a library is a directory or the license is empty", async () => {
    const fixture = await fixtureArchive()
    const { server, url } = serve(fixture.archive)
    try {
      const cache = join(fixture.root, "cache")
      const redistributable = pin(url, fixture.bytes.byteLength, fixture.sha256, fixture.archiveRoot)
      const first = await Effect.runPromise(stageNvrtc(redistributable, cache).pipe(Effect.provide(BunContext.layer)))
      await rm(first.libraries[1]!)
      await mkdir(first.libraries[1]!)
      await writeFile(first.license, "")

      const repaired = await Effect.runPromise(stageNvrtc(redistributable, cache).pipe(Effect.provide(BunContext.layer)))
      expect(repaired).toEqual(first)
      expect(await readFile(repaired.libraries[1]!, "utf8")).toBe("builtins")
      expect(await readFile(repaired.license, "utf8")).toBe("license")
      expect(await readdir(cache)).toEqual([fixture.sha256])
    } finally {
      server.stop(true)
    }
  })
})

describe("NVRTC release pins", () => {
  it("pins NVRTC 12.9 for every CUDA host and none for Apple hosts", () => {
    const pinned = releaseHosts.filter((host) => Option.isSome(host.nvrtc)).map((host) => host.id)
    expect(pinned).toEqual(["linux-arm64-gnu", "linux-x64-gnu", "windows-x64-msvc"])
    const names = (id: Parameters<typeof hostById>[0]) =>
      Option.getOrThrow(hostById(id).nvrtc).libraries.map((file) => file.name)
    expect(names("linux-x64-gnu")).toEqual(["libnvrtc.so.12", "libnvrtc-builtins.so.12.9"])
    expect(names("linux-arm64-gnu")).toEqual(["libnvrtc.so.12", "libnvrtc-builtins.so.12.9"])
    expect(names("windows-x64-msvc")).toEqual(["nvrtc64_120_0.dll", "nvrtc-builtins64_129.dll"])
    for (const host of releaseHosts) {
      Option.map(host.nvrtc, (nvrtc) => {
        expect(nvrtc.version).toBe("12.9.86")
        expect(nvrtc.url.startsWith("https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/")).toBe(true)
        expect(nvrtc.sha256).toMatch(/^[a-f0-9]{64}$/)
        expect([...nvrtc.libraries, nvrtc.license].every((file) => !file.member.includes(".alt"))).toBe(true)
      })
    }
  })
})
