import { FetchHttpClient, FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { Effect, Layer, Option, Schema, Stream } from "effect"
import { expect, test } from "vitest"
import { join } from "node:path"
import releasePlan from "../../release/release-plan.json"
import { acquireRelease, releaseUrl } from "../../release/src/acquisition"
import { defaultArtifactDownloadPolicy, downloadArtifact } from "../../release/src/artifact-download"
import { ArtifactStore, fileArtifactStore } from "../src/artifact-store"
import { runtimeEnvironment, runtimeRelease } from "../src/runtime-release"
import { sha256 } from "../src/snapshot"

const payload = new TextEncoder().encode("unpublished native archive fixture")
const manifest = Schema.decodeUnknownSync(ReleaseManifestSchema)({ schemaVersion: 2, version: "0.1.3", acnRevision: 1,
  rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.3", sourceCommit: "a".repeat(40), artifacts: [{
    id: "icn-base-darwin-arm64", kind: "icn-base", host: "darwin-arm64", backend: "cpu", filename: "native-base.tar.gz",
    nativeBuild: "native-fixture", backendModuleAbi: "fixture-abi", bytes: payload.length, sha256: sha256(payload),
  }] })

test("ordinary acquisition reads the admitted release and only its host runtime routes", async () => {
  let endpoint = ""
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-runtime-test-" })
    yield* Effect.gen(function* () {
      const objects = yield* ArtifactStore
      yield* objects.put(sha256(payload), Stream.make(payload))
      const server = yield* runtimeRelease(manifest, "darwin-arm64")
      endpoint = server.baseUrl
      const acquired = yield* acquireRelease(server.baseUrl, manifest.version, join(root, "cache"))
      expect(acquired.manifest).toEqual(manifest)
      const url = releaseUrl(server.baseUrl, manifest.version, manifest.artifacts[0]!.filename)
      const response = yield* Effect.tryPromise(() => fetch(url))
      expect(response.status).toBe(200)
      expect(response.headers.get("content-length")).toBe(String(payload.length))
      expect(yield* Effect.tryPromise(() => response.text())).toBe(new TextDecoder().decode(payload))
      const destination = join(root, "downloaded-runtime")
      yield* downloadArtifact({ url, destination, bytes: payload.length, sha256: sha256(payload),
        strategy: { _tag: "Segmented", concurrency: 4, chunkBytes: 8, fallbackToSequential: false }, policy: defaultArtifactDownloadPolicy,
        onProgress: Option.none(), onVerificationProgress: Option.none() })
      expect(Array.from(yield* fs.readFile(destination))).toEqual(Array.from(payload))
      const invalidRange = yield* Effect.tryPromise(() => fetch(url, { headers: { range: "bytes=9999-10000" } }))
      expect(invalidRange.status).toBe(416)
      // Mutating the CAS after preparation cannot change the published private copy.
      yield* fs.writeFileString(join(root, "objects", sha256(payload)), "corrupted later")
      expect(yield* Effect.tryPromise(() => fetch(url).then(response => response.text()))).toBe(new TextDecoder().decode(payload))
      for (const bad of [releaseUrl(server.baseUrl, "0.1.4", manifest.artifacts[0]!.filename), `${url}?alternate=1`,
        releaseUrl(server.baseUrl, manifest.version, "not-admitted.tar.gz"), new URL("/magnitude-release.json", endpoint).href]) {
        expect((yield* Effect.tryPromise(() => fetch(bad))).status).toBe(404)
      }
      expect((yield* Effect.tryPromise(() => fetch(url, { method: "POST" }))).status).toBe(404)
    }).pipe(Effect.provide(fileArtifactStore(join(root, "objects"))))
  })).pipe(Effect.provide(Layer.merge(BunContext.layer, FetchHttpClient.layer))))
  await expect(fetch(releaseUrl(endpoint, manifest.version, "magnitude-release.json"))).rejects.toThrow()
})

test("missing host runtimes and corrupt admitted bytes fail before publication", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-runtime-rejection-" })
  yield* Effect.gen(function* () {
    const objects = yield* ArtifactStore
    yield* objects.put(sha256(payload), Stream.make(payload))
    expect((yield* runtimeRelease(manifest, "linux-x64-gnu").pipe(Effect.either))._tag).toBe("Left")
    expect((yield* runtimeRelease({ ...manifest, artifacts: [{ ...manifest.artifacts[0]!, filename: "magnitude-release.json" }] }, "darwin-arm64").pipe(Effect.either))._tag).toBe("Left")
    expect((yield* runtimeRelease({ ...manifest, artifacts: [{ ...manifest.artifacts[0]!, bytes: 1 }] }, "darwin-arm64").pipe(Effect.either))._tag).toBe("Left")
    yield* fs.writeFileString(join(root, "objects", sha256(payload)), "corrupt")
    expect((yield* runtimeRelease(manifest, "darwin-arm64").pipe(Effect.either))._tag).toBe("Left")
  }).pipe(Effect.provide(fileArtifactStore(join(root, "objects"))))
})).pipe(Effect.provide(BunContext.layer))))

test("candidate and baseline runtime environments remain independent across baseline cleanup", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-runtime-pair-" })
  yield* Effect.gen(function* () {
    const objects = yield* ArtifactStore
    yield* objects.put(sha256(payload), Stream.make(payload))
    const environment = { HOME: root, MAGNITUDE_ICN_PATH: "/ambient/development/installation.json", magnitude_icn_path: "/another/ambient/path",
      MAGNITUDE_RELEASE_BASE_URL: "https://unrelated.invalid", magnitude_release_base_url: "https://another.invalid" }
    const appOnly: typeof ReleaseManifestSchema.Type = { ...manifest, artifacts: [{ ...manifest.artifacts[0]!, id: "desktop-darwin-arm64", kind: "desktop", filename: "Magnitude.dmg",
      backend: Option.none(), nativeBuild: Option.none(), backendModuleAbi: Option.none() }] }
    const ordinary = yield* runtimeEnvironment(appOnly, "darwin-arm64", environment)
    expect(ordinary.MAGNITUDE_ICN_PATH).toBeUndefined()
    expect(ordinary.magnitude_icn_path).toBeUndefined()
    const candidate = yield* runtimeEnvironment(manifest, "darwin-arm64", environment)
    expect(candidate.MAGNITUDE_ICN_PATH).toBeUndefined()
    expect(candidate.magnitude_icn_path).toBeUndefined()
    expect(candidate.magnitude_release_base_url).toBeUndefined()
    expect(candidate.HOME).toBe(root)
    let baselineUrl = ""
    yield* Effect.scoped(Effect.gen(function* () {
      const previous = { ...manifest, version: "0.1.2", tag: "@magnitudedev/cli@0.1.2" }
      const baseline = yield* runtimeEnvironment(previous, "darwin-arm64", environment)
      baselineUrl = baseline.MAGNITUDE_RELEASE_BASE_URL!
      expect(baselineUrl).not.toBe(candidate.MAGNITUDE_RELEASE_BASE_URL)
      expect(baseline.MAGNITUDE_ICN_PATH).toBeUndefined()
      const acquired = yield* acquireRelease(baselineUrl, previous.version, join(root, "baseline-cache"))
      expect(acquired.manifest.version).toBe(previous.version)
      expect((yield* Effect.tryPromise(() => fetch(releaseUrl(baselineUrl, manifest.version, "magnitude-release.json")))).status).toBe(404)
    }))
    expect((yield* Effect.tryPromise(() => fetch(releaseUrl(baselineUrl, "0.1.2", "magnitude-release.json"))).pipe(Effect.either))._tag).toBe("Left")
    expect((yield* acquireRelease(candidate.MAGNITUDE_RELEASE_BASE_URL!, manifest.version, join(root, "candidate-cache"))).manifest.version).toBe(manifest.version)
    expect(environment.MAGNITUDE_ICN_PATH).toBe("/ambient/development/installation.json")
  }).pipe(Effect.provide(fileArtifactStore(join(root, "objects"))))
})).pipe(Effect.provide(Layer.merge(BunContext.layer, FetchHttpClient.layer)))))

test("one inherited runtime origin serves both update versions without aliasing their manifests", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-update-runtime-" })
  yield* Effect.gen(function* () {
    const objects = yield* ArtifactStore
    yield* objects.put(sha256(payload), Stream.make(payload))
    const next = { ...manifest, version: "0.1.4", tag: "@magnitudedev/cli@0.1.4" }
    const environment = yield* runtimeEnvironment(manifest, "darwin-arm64", {}, [next])
    const origin = environment.MAGNITUDE_RELEASE_BASE_URL!
    for (const value of [manifest, next]) {
      const acquired = yield* acquireRelease(origin, value.version, join(root, `cache-${value.version}`))
      expect(acquired.manifest).toEqual(value)
      expect(yield* Effect.tryPromise(() => fetch(releaseUrl(origin, value.version, value.artifacts[0]!.filename)).then(response => response.text()))).toBe(new TextDecoder().decode(payload))
    }
    expect((yield* Effect.tryPromise(() => fetch(releaseUrl(origin, "0.1.5", "magnitude-release.json")))).status).toBe(404)
    expect((yield* runtimeRelease(manifest, "darwin-arm64", [manifest]).pipe(Effect.either))._tag).toBe("Left")
    expect((yield* runtimeRelease(manifest, "darwin-arm64", [{ ...next, artifacts: [{ ...next.artifacts[0], host: Option.some("linux-x64-gnu" as const) }] }]).pipe(Effect.either))._tag).toBe("Left")
  }).pipe(Effect.provide(fileArtifactStore(join(root, "objects"))))
})).pipe(Effect.provide(Layer.merge(BunContext.layer, FetchHttpClient.layer)))))
